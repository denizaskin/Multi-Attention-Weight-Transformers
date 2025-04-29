import os
# Do not disable the MPS memory allocation cap.
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Removed for MPS stability

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Suppress checkpoint warning.
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

# Auto-detect best device.
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Use FP16 only when running on CUDA to avoid numerical instabilities on MPS.
use_fp16 = device.type == "cuda"

# Global variable use_grpo will be defined in main.
# Global testing configuration (for fixed depth index) is defined in main.

# Define GRPO-based Depth Selector.
class GRPODepthSelector(nn.Module):
    def __init__(self, depth_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(depth_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, depth_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, attn_5d):
        batch_size = attn_5d.shape[0]
        # Collapse the last two dims for a summary.
        x = attn_5d.mean(dim=(-2, -3)).view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        if use_grpo:
            depth_probs = self.softmax(logits)
            depth_index = torch.multinomial(depth_probs, num_samples=1).squeeze(-1)
            depth_routing = F.one_hot(depth_index, num_classes=logits.shape[-1]).float()
        else:
            # Use Gumbel Softmax reparameterization (differentiable).
            depth_probs = F.gumbel_softmax(logits, tau=1.0, hard=False)
            depth_index = depth_probs.argmax(dim=-1)
            depth_routing = depth_probs  # Use soft routing.
        return depth_routing, depth_probs, depth_index

# Define Learnable 5D Attention Layer with GRPO using vectorized depth transforms.
class ModifiedMistralAttention(nn.Module):
    def __init__(self, original_attention, num_heads=32, head_dim=128, depth_dim=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.depth_dim = depth_dim
        self.depth_selector = GRPODepthSelector(depth_dim)
        # Combine depth-specific transforms into one parameter.
        self.depth_weights = nn.Parameter(torch.empty(depth_dim, head_dim, head_dim))
        nn.init.xavier_uniform_(self.depth_weights)
        self.original_attention = original_attention
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, Q, K, V):
        # Remove cloning/detaching so gradients remain connected.
        batch_size, seq_len_q, num_heads, head_dim = Q.shape
        seq_len_k = K.shape[1]
        # Reshape Q and K for 5D attention.
        Q = Q.view(batch_size, num_heads, seq_len_q, head_dim, 1)
        K = K.view(batch_size, num_heads, seq_len_k, 1, head_dim)
        attn_5d = torch.matmul(Q, K) * self.scale
        # This reshaping assumes the two inner dims correspond to depth-related info.
        attn_5d = attn_5d.squeeze(-1).permute(0, 1, 3, 4, 2)  # Shape: (B, num_heads, head_dim, ?, seq_len_k)

        # Compute transformed values (incorporating depth dimension)
        V_reordered = V.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
        # transformed_values: (B, seq_len, num_heads, head_dim, depth_dim)
        transformed_values = torch.einsum("bhqi, dij -> bhqjd", V_reordered, self.depth_weights)
        transformed_values = transformed_values.permute(0, 2, 1, 3, 4)

        if self.training:
            depth_routing, depth_probs, depth_index = self.depth_selector(attn_5d)
            selected_V = torch.einsum("bkhvd, bkd -> bkhv", transformed_values, depth_routing)
            attn_weights = torch.einsum("bhqkd, bkd -> bhqk", attn_5d, depth_routing)
        else:
            # In testing, use the fixed depth index to avoid extra computation.
            selected_V = transformed_values[..., user_depth_index]  # (B, seq_len, num_heads, head_dim)
            selected_V = selected_V.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
            attn_weights = attn_5d[..., user_depth_index]  # select along the depth dim
            attn_weights = attn_weights.permute(0, 1, 3, 2)
            depth_probs = torch.empty(0, device=Q.device)
            depth_index = torch.empty(0, device=Q.device, dtype=torch.long)

        attn_output = torch.einsum("bhqk, bkhv -> bqhv", attn_weights, selected_V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_q, -1)
        if self.training:
            return attn_output, depth_probs, depth_index
        else:
            return attn_output, torch.empty(0, device=attn_output.device), torch.empty(0, device=attn_output.device, dtype=torch.long)

# Define Custom Trainer with GRPO Loss.
class CustomTrainer(Trainer):
    def grpo_loss_function(self, depth_probs, depth_index, ce_loss_per_sample):
        reward = -ce_loss_per_sample
        baseline = reward.mean()
        log_probs = torch.log(depth_probs + 1e-8)
        chosen_log_probs = log_probs.gather(1, depth_index.unsqueeze(1)).squeeze()
        policy_loss = -chosen_log_probs * (reward - baseline)
        entropy = -torch.sum(depth_probs * log_probs, dim=-1).mean()
        grpo_loss = policy_loss.mean() - 0.1 * entropy
        return grpo_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        if isinstance(outputs, tuple) and len(outputs) == 3:
            logits, depth_probs, depth_index = outputs
            ce_loss_per_sample = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=-100, reduction='none'
            ).view(logits.size(0), -1).mean(dim=1)
            ce_loss = ce_loss_per_sample.mean()
            if model.training and self.use_grpo:
                grpo_loss = self.grpo_loss_function(depth_probs, depth_index, ce_loss_per_sample)
                loss = ce_loss + 0.1 * grpo_loss
            else:
                loss = ce_loss
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_fp16):
            loss = self.compute_loss(model, inputs)
        loss_tensor = loss.loss if not isinstance(loss, torch.Tensor) and hasattr(loss, "loss") else loss
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        if self._step_count % 10 == 0:
            if use_grpo:
                print("GRPO Loss:", loss_tensor.item())
            else:
                print("Gumbel Softmax Loss:", loss_tensor.item())
            if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        return loss

def load_model(model_name, use_fp16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_map = "auto" if use_fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map=device_map
    )
    if device_map is None:
        model = model.to(device)
    if device.type == "mps":
        if any(param.device.type == "meta" for param in model.parameters()):
            print("⚠️ Some parameters are offloaded; skipping memory_format conversion.")
        else:
            model = model.to(memory_format=torch.channels_last)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled.")
        model.config.use_cache = False
    return model, tokenizer

def modify_model_attention(model):
    target_modules = [(name, module) for name, module in model.named_modules()]
    layer_indices = [int(name.split(".")[2]) for name, _ in target_modules 
                     if "model.layers." in name and name.split(".")[2].isdigit()]
    last_layer_index = max(layer_indices) if layer_indices else None
    if last_layer_index is None:
        raise ValueError("No valid transformer layers found in the model!")
    last_layer_name = f"model.layers.{last_layer_index}.self_attn"
    for name, module in target_modules:
        if name == last_layer_name:
            module.k_proj.weight = nn.Parameter(module.k_proj.weight.clone().detach().requires_grad_(True))
            module.v_proj.weight = nn.Parameter(module.v_proj.weight.clone().detach().requires_grad_(True))
            new_module = ModifiedMistralAttention(module)
            new_module.original_attention = None
            if device.type != "mps":
                try:
                    new_module = torch.jit.script(new_module)
                    print("✅ Scripted the modified attention module for optimized performance!")
                except Exception as e:
                    print("⚠️ Could not script the modified attention module:", e)
            else:
                print("Running on MPS: skipping torch.jit.script for stability.")
            setattr(model, name, new_module)
    print(f"✅ Modified the last attention layer: {last_layer_name}")
    return model, last_layer_index

def freeze_layers_except_last(model, last_layer_index):
    for name, param in model.named_parameters():
        if f"model.layers.{last_layer_index}" not in name:
            param.requires_grad = False
    return model

def prepare_datasets(tokenizer, dataset_name):
    dataset = load_dataset(dataset_name, split="test").shuffle(seed=42)
    if len(dataset) < 350:
        raise ValueError("Dataset does not have enough examples for partitioning into training and held-out test sets.")
    train_dataset = dataset.select(range(300))
    held_out_test_dataset = dataset.select(range(300,350))
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["challenge"], truncation=True, padding="max_length", max_length=128)
        labels = tokenizer(examples["answer"], truncation=True, padding="max_length", max_length=128)
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels["input_ids"]
        ]
        inputs["labels"] = labels["input_ids"]
        return inputs

    num_proc = 1 if device.type == "mps" else (os.cpu_count() if os.cpu_count() is not None else 1)
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    held_out_test_dataset = held_out_test_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    return train_dataset, held_out_test_dataset

def train_model(model, train_dataset, training_args):
    trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset)
    print("🚀 Fine-tuning DeepSeek LLM with GRPO Depth Selection...")
    trainer.train()
    print(f"✅ Fine-tuning complete! Saving model to {training_args.output_dir}...")
    model.save_pretrained(training_args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"✅ Model saved to {training_args.output_dir}!")

def generate_text(model, tokenizer, prompt):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_fp16):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print("\n🚀 Model Output: ", end="", flush=True)
        generated_ids = model.generate(
            **inputs,
            max_length=800,
            temperature=0.9,
            num_beams=1,  # Reduced from 5 to 1 for faster decoding
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        for token_id in generated_ids[0][len(inputs["input_ids"][0]):]:
            token = tokenizer.decode(token_id, skip_special_tokens=True)
            print(token, end="", flush=True)
        print()

def answer_holdout_dataset(model, tokenizer, held_out_test_dataset):
    model.eval()
    for i, sample in enumerate(held_out_test_dataset):
        prompt = sample["challenge"]
        print(f"\nQuestion {i+1}: {prompt}")
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_fp16):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(
                **inputs,
                max_length=800,
                temperature=0.9,
                num_beams=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
        answer = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"Answer {i+1}: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    global use_attention_index, user_depth_index
    use_attention_index = False      # This flag is no longer used in testing branch.
    user_depth_index = 4            # User-specified index (an integer in [0, depth_dim-1]).

    use_grpo = False  # Set False to use Gumbel Softmax variant.
    training = False  
    checkpoint_path = "./deepseek_finetued-reinforcement3"
    dataset_name = "nuprl/verbal-reasoning-challenge"
    "Qwen/Qwen2.5-1.5B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model, tokenizer = load_model(model_name, use_fp16)

    model, last_layer_index = modify_model_attention(model)
    model = freeze_layers_except_last(model, last_layer_index)

    if device.type != "mps":
        try:
            model = torch.compile(model)
            print("✅ Model compiled with torch.compile for optimized performance!")
        except Exception as e:
            print("⚠️ torch.compile not available or failed:", e)
    else:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        print("⚠️ Running on MPS: torch.compile errors suppressed; using eager mode.")

    train_dataset, held_out_test_dataset = prepare_datasets(tokenizer, dataset_name)
    import random
    held_out_test_prompt = held_out_test_dataset[random.randint(0, len(held_out_test_dataset)-1)]["challenge"]

    if training:
        train_model(model, train_dataset, TrainingArguments(
            output_dir=checkpoint_path,
            per_device_train_batch_size=5,
            gradient_accumulation_steps=5,
            num_train_epochs=20,
            logging_dir="./logs",
            save_strategy="epoch",
            learning_rate=2e-5,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        ))
    print("\nUsing held-out test question for single inference:")
    generate_text(model, tokenizer, "What letter comes next in this series: W, L, C, N, I, T?")
    
    # Uncomment below to iterate over the held-out test dataset.
    # answer_holdout_dataset(model, tokenizer, held_out_test_dataset)