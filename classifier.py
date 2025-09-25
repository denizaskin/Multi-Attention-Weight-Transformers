from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

def load_imdb_dataset(sample_fraction=0.05):
    print("DEBUG: Loading IMDB dataset from Hugging Face...")
    dataset = load_dataset("glue", "sst2")
    
    # Calculate the number of samples to take
    total_samples = len(dataset["train"]) + len(dataset["validation"])
    sample_size = int(total_samples * sample_fraction)
    
    # Get the actual sizes of train and validation sets
    train_size = len(dataset["train"])
    validation_size = len(dataset["validation"])
    
    # Calculate proportional sample sizes based on original dataset proportions
    train_fraction = train_size / total_samples
    train_sample_size = min(int(sample_size * train_fraction), train_size)
    validation_sample_size = min(sample_size - train_sample_size, validation_size)
    
    print(f"DEBUG: Sampling {train_sample_size} from train and {validation_sample_size} from validation")
    
    # Combine train and test splits and sample directly
    from datasets import concatenate_datasets
    train_sample = dataset["train"].shuffle(seed=42).select(range(train_sample_size))
    validation_sample = dataset["validation"].shuffle(seed=42).select(range(validation_sample_size))
    combined_dataset = concatenate_datasets([train_sample, validation_sample])
    
    # Extract text and labels
    X = np.array(combined_dataset["sentence"])
    y = np.array(combined_dataset["label"])
    
    # Split into train/test with 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nDataset sizes:")
    print(f"Original size: {total_samples}")
    print(f"Sampled size ({int(sample_fraction * 100)}%): {len(X)}")
    print(f"Training size (80% of sampled): {len(X_train)}")
    print(f"Testing size (20% of sampled): {len(X_test)}")
    
    # Return dataset sizes along with the data
    dataset_sizes = {
        'total': total_samples,
        'sampled': len(X),
        'train': len(X_train),
        'test': len(X_test)
    }
    
    return X_train, X_test, y_train, y_test, dataset_sizes

def load_sst2_dataset(sample_fraction):
    print("DEBUG: Loading SST-2 dataset from Hugging Face...")
    dataset = load_dataset("glue", "sst2")
    
    # Combine train and validation splits
    X = np.array(dataset["train"]["sentence"] + dataset["validation"]["sentence"])
    y = np.array(dataset["train"]["label"] + dataset["validation"]["label"])
    
    # Take a fraction of the data
    total_samples = len(X)
    sample_size = int(total_samples * sample_fraction)
    indices = np.random.choice(total_samples, sample_size, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    
    # Split into train/test with 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_sampled
    )
    
    print(f"\nDataset sizes:")
    print(f"Original size: {total_samples}")
    print(f"Sampled size ({int(sample_fraction * 100)}%): {sample_size}")
    print(f"Training size (80% of sampled): {len(X_train)}")
    print(f"Testing size (20% of sampled): {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def get_gpt4_analysis(metrics_dict, dataset_info):
    try:
        client = OpenAI(timeout=30.0)  # Add timeout of 30 seconds
        
        prompt = f"""
        Please analyze the performance of a sentiment classification model on the IMDB movie reviews dataset.

        Dataset Information:
        - Task: Binary sentiment classification (positive/negative) of movie reviews
        - Original dataset size: {dataset_info['total']} reviews
        - Sample used: {dataset_info['sampled']} reviews (25% of original)
        - Training set: {dataset_info['train']} reviews
        - Test set: {dataset_info['test']} reviews

        Model Performance Metrics:
        - Precision: {metrics_dict['precision']:.4f}
        - Recall: {metrics_dict['recall']:.4f}
        - F1 Score (non-weighted): {metrics_dict['f1']:.4f}
        - F1 Score (weighted): {metrics_dict['f1_weighted']:.4f}

        Please provide:
        1. An interpretation of these metrics
        2. Analysis of model performance
        3. Potential areas for improvement
        """

        print("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "API request failed. Please check your API key and internet connection."

def main():
    print("DEBUG: Starting main function")
    # Load dataset
    print("DEBUG: Loading IMDB dataset...")
    X_train, X_test, y_train, y_test, dataset_sizes = load_imdb_dataset(sample_fraction=0.10)
    
    # Print dataset information
    print("\nDetailed Dataset Information:")
    print("=" * 50)
    print(f"Total dataset size: {len(X_train) + len(X_test)} reviews")
    print(f"Training set size: {len(X_train)} reviews ({len(X_train)/(len(X_train) + len(X_test))*100:.1f}%)")
    print(f"Testing set size:  {len(X_test)} reviews ({len(X_test)/(len(X_train) + len(X_test))*100:.1f}%)")
    print("=" * 50)
    
    # Create TF-IDF vectorizer
    print("\nVectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train logistic regression
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Print results
    print("\nResults:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score (non-weighted): {f1:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"Unique classes in y_test: {np.unique(y_test)}")
    print(f"Unique classes in y_pred: {np.unique(y_pred)}")

    # Prepare dataset info for GPT-4 (modified)
    dataset_info = dataset_sizes
    
    # Prepare metrics for GPT-4
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_weighted': f1_weighted
    }

    # Get GPT-4 analysis
    print("\nGetting GPT-4 analysis...")
    analysis = get_gpt4_analysis(metrics_dict, dataset_info)
    
    print("\nGPT-4 Analysis:")
    print("=" * 80)
    print(analysis)
    print("=" * 80)

# At the beginning of main()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    masked_key = f"{api_key[:5]}...{api_key[-4:]}"
    print(f"API key loaded: {masked_key}")
else:
    print("No API key found!")

if __name__ == "__main__":
    main()