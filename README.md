# MAWT: Multi-Attention-Weight Transformer

## 🧠 Overview

The **Multi-Attention-Weight Transformer (MAWT)** introduces a novel enhancement to the traditional Transformer architecture by incorporating an additional **depth dimension** into the self-attention mechanism. This modification transforms the standard attention tensor from a 4D structure:

```
(batch_size, num_heads, seq_len_query, seq_len_key)
```


to a 5D structure:

```
(batch_size, num_heads, seq_len_query, seq_len_key, depth)
```


This added dimension allows the model to capture more nuanced relationships within the data, enabling a richer and more flexible representation of context.

## 🔍 Key Features

- **Depth-Enhanced Self-Attention**: By extending the attention mechanism to include a depth dimension, MAWT enables the model to process information across multiple representational layers simultaneously. This facilitates a more comprehensive understanding of complex patterns and dependencies within the data.

- **Dynamic Depth Selection**: MAWT employs a **Group Relative Policy Optimization (GRPO)**-based selector to dynamically determine the most relevant depth slices for each input. During training, this selector learns to focus on the depth dimensions that contribute most significantly to the model's performance, allowing for adaptive and context-sensitive attention.

- **Modified Attention Mechanism**: The traditional attention computation is augmented to handle the additional depth dimension. This involves learnable transformations that operate across the depth slices, enabling the model to integrate information from multiple depths effectively.

- **GRPO Fine-Tuning**: The integration of GRPO allows for reinforcement learning-based fine-tuning of the depth selection strategy. By treating the selection of depth slices as actions and optimizing for performance-based rewards, the model learns to make more informed and effective depth selections over time.

## 🛠️ Components

### 1. GRPO-Based Depth Selector

The `GRPODepthSelector` module analyzes the 5D attention tensor to compute probabilities over the depth dimension. During training, it samples depth indices based on these probabilities, allowing the model to explore various depth combinations. During inference, it selects the most probable depth slice, ensuring consistent and optimized performance.

### 2. Modified Mistral Attention Layer

The `ModifiedMistralAttention` layer extends the standard attention mechanism to operate over the additional depth dimension. It applies learnable transformations across depth slices, enabling the model to capture complex interactions and dependencies that span multiple representational layers.

### 3. Custom Trainer with GRPO Loss

The `CustomTrainer` class incorporates a GRPO-based loss function that combines cross-entropy loss with a policy loss derived from the depth selection probabilities. This approach encourages the model to learn effective depth selection strategies that enhance overall performance.

## 📊 Training and Evaluation

- **Dataset**: The model is trained on the "nuprl/verbal-reasoning-challenge" dataset, which presents tasks requiring advanced reasoning capabilities.

- **Model**: MAWT builds upon the "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" model, leveraging its robust architecture as a foundation for the enhanced attention mechanism.

- **Training Strategy**: The model undergoes fine-tuning using the GRPO approach, allowing it to learn optimal depth selection strategies that improve its reasoning and decision-making abilities.

## 🚀 Getting Started

1. **Load the Model**: Initialize the base model and tokenizer from the specified pre-trained model.

2. **Modify Attention**: Replace the last attention layer with the `ModifiedMistralAttention` layer to enable dynamic depth selection.

3. **Freeze Layers**: Freeze all layers except the last one to focus training on the modified attention mechanism.

4. **Prepare Dataset**: Load and preprocess the dataset, ensuring it's properly tokenized and formatted for training.

5. **Train the Model**: Fine-tune the model using the `CustomTrainer`, which incorporates the GRPO loss function to guide the learning process.

6. **Evaluate Performance**: Assess the model's performance on held-out test data to measure improvements in reasoning capabilities.

## 📈 Results

The integration of a depth dimension and dynamic depth selection in MAWT has demonstrated enhanced performance in reasoning tasks. By allowing the model to adaptively focus on different representational layers, it captures more complex patterns and dependencies, leading to more accurate and insightful responses.

## 🧪 Future Work

- **Expand Depth Dimensions**: Explore the impact of increasing the number of depth slices to capture even richer representations.

- **Broader Applications**: Apply the MAWT architecture to other domains, such as code generation or mathematical problem-solving, to assess its versatility.

- **Enhanced Training Techniques**: Investigate alternative reinforcement learning strategies to further improve the model's depth selection capabilities.

---

*Note: The MAWT architecture builds upon concepts introduced in the DeepSeekMath paper, which explores the use of GRPO for enhancing mathematical reasoning in language models.*

--- 
