#!/usr/bin/env python3
"""
Privacy Policy Analysis Model Training Script
Uses the real OPP-115 dataset from sanitized_policies folder
"""

import os
import json
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO
import pickle
import warnings
import glob
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import datasets
from datasets import Dataset

def download_opp115_dataset():
    """Download and extract the OPP-115 dataset"""
    print("Downloading OPP-115 dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # OPP-115 dataset URL
    url = "https://www.usableprivacy.org/static/data/OPP-115_v1_0.zip"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Extract the zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall('data/')
        
        print("Dataset downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_real_opp115_data():
    """Load and process the real OPP-115 dataset from sanitized_policies"""
    print("Loading real OPP-115 dataset...")
    
    # Check multiple possible paths
    possible_paths = [
        "data/OPP-115/sanitized_policies",
        "data/sanitized_policies", 
        "OPP-115/sanitized_policies",
        "sanitized_policies"
    ]
    
    sanitized_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sanitized_path = path
            print(f"Found sanitized policies at: {path}")
            break
    
    if not sanitized_path:
        print(f"Sanitized policies directory not found. Checked paths: {possible_paths}")
        # List what's actually in the data directory
        if os.path.exists("data"):
            print("Contents of data directory:")
            for item in os.listdir("data"):
                print(f"  - {item}")
                if os.path.isdir(os.path.join("data", item)):
                    sub_items = os.listdir(os.path.join("data", item))
                    for sub_item in sub_items:
                        print(f"    - {sub_item}")
        return None
    
    # Get all HTML files
    policy_files = glob.glob(os.path.join(sanitized_path, "*.html"))
    print(f"Found {len(policy_files)} HTML policy files")
    
    if not policy_files:
        print("No HTML policy files found!")
        return None
    
    all_texts = []
    all_labels = []
    
    from bs4 import BeautifulSoup
    
    for policy_file in policy_files:
        try:
            with open(policy_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Split into sentences/paragraphs for better training
            sentences = clean_text.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip if too short
                if len(sentence) < 50:
                    continue
                
                # Create labels based on text content analysis
                labels = create_labels_from_text(sentence)
                
                all_texts.append(sentence)
                all_labels.append(labels)
                
                # Stop if we have enough data for training
                if len(all_texts) >= 1000:
                    break
        
        except Exception as e:
            print(f"Error processing {policy_file}: {e}")
            continue
        
        if len(all_texts) >= 1000:
            break
    
    if not all_texts:
        print("No text data extracted from policy files!")
        return None
    
    # Create DataFrame
    df_data = {'text': all_texts}
    for key in ['data_collection', 'data_sharing', 'data_storage', 'vague_language']:
        df_data[key] = [label[key] for label in all_labels]
    
    df = pd.DataFrame(df_data)
    print(f"Successfully loaded {len(df)} text segments from real OPP-115 data")
    
    # Print label distribution
    for label in ['data_collection', 'data_sharing', 'data_storage', 'vague_language']:
        count = df[label].sum()
        print(f"{label}: {count} positive samples ({count/len(df)*100:.1f}%)")
    
    return df

def load_annotations_data():
    """Load annotation data if available"""
    annotations_path = "data/OPP-115/annotations"
    
    if not os.path.exists(annotations_path):
        print("Annotations directory not found, using text-based labeling only")
        return {}
    
    annotations = {}
    annotation_files = glob.glob(os.path.join(annotations_path, "*.json"))
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
                
            # Extract filename without extension as key
            file_key = os.path.basename(ann_file).replace('.json', '')
            annotations[file_key] = ann_data
            
        except Exception as e:
            print(f"Error loading annotations from {ann_file}: {e}")
            continue
    
    print(f"Loaded annotations for {len(annotations)} files")
    return annotations

def create_labels_from_text(text):
    """Create labels based on text content analysis"""
    text_lower = text.lower()
    
    labels = {
        'data_collection': 0,
        'data_sharing': 0,
        'data_storage': 0,
        'vague_language': 0
    }
    
    # Data collection keywords
    collection_keywords = [
        'collect', 'gather', 'obtain', 'acquire', 'receive', 'record', 'capture',
        'personal information', 'personal data', 'user data', 'information about you',
        'your information', 'collect information', 'data we collect'
    ]
    
    # Data sharing keywords
    sharing_keywords = [
        'share', 'disclose', 'provide to', 'transfer', 'third party', 'third parties',
        'partners', 'affiliates', 'service providers', 'share with', 'disclose to',
        'provide your information', 'share your data'
    ]
    
    # Data storage keywords
    storage_keywords = [
        'store', 'retain', 'keep', 'maintain', 'preserve', 'hold', 'save',
        'data storage', 'information storage', 'security', 'protect', 'secure',
        'encryption', 'safeguard', 'data security'
    ]
    
    # Vague language keywords
    vague_keywords = [
        'may', 'might', 'could', 'possible', 'reasonable', 'appropriate',
        'necessary', 'discretion', 'from time to time', 'as needed',
        'business purposes', 'legitimate interests', 'improve services',
        'enhance user experience', 'other purposes', 'similar purposes',
        'as we deem', 'in our sole discretion', 'we reserve the right'
    ]
    
    # Check for each category
    for keyword in collection_keywords:
        if keyword in text_lower:
            labels['data_collection'] = 1
            break
    
    for keyword in sharing_keywords:
        if keyword in text_lower:
            labels['data_sharing'] = 1
            break
    
    for keyword in storage_keywords:
        if keyword in text_lower:
            labels['data_storage'] = 1
            break
    
    for keyword in vague_keywords:
        if keyword in text_lower:
            labels['vague_language'] = 1
            break
    
    return labels

def create_multi_label_classifier(df):
    """Create and train a multi-label classification model"""
    print("Creating multi-label classifier...")
    print(f"Training on {len(df)} samples")
    
    # Prepare features and labels
    texts = df['text'].tolist()
    labels = df[['data_collection', 'data_sharing', 'data_storage', 'vague_language']].values.astype(np.float32)
    
    # Check for any issues with labels
    print(f"Label shape: {labels.shape}")
    print(f"Label distribution:")
    for i, col in enumerate(['data_collection', 'data_sharing', 'data_storage', 'vague_language']):
        print(f"  {col}: {np.sum(labels[:, i])} positive samples")
    
    # Ensure we have some positive samples for each class
    for i in range(labels.shape[1]):
        if np.sum(labels[:, i]) == 0:
            # Add at least one positive sample for each class to avoid training issues
            labels[i, i] = 1.0
    
    # Split data - use simple split if stratification fails
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        # If stratification fails, use simple split
        print("Stratification failed, using simple split")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize data with consistent parameters - KEY FIX: Don't return tensors here
    max_length = 256
    train_encodings = tokenizer(
        X_train, 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )
    test_encodings = tokenizer(
        X_test, 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )
    
    # Create custom dataset class - FIXED tensor handling
    class PrivacyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels, dtype=torch.float32)
        
        def __getitem__(self, idx):
            # Convert to tensors only when accessing individual items
            item = {}
            for key, val in self.encodings.items():
                item[key] = torch.tensor(val[idx], dtype=torch.long)
            item['labels'] = self.labels[idx]
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = PrivacyDataset(train_encodings, y_train)
    test_dataset = PrivacyDataset(test_encodings, y_test)
    
    # Load model for multi-label classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,  # 4 categories
        problem_type="multi_label_classification"
    )
    
    # Resize token embeddings if we added a pad token
    model.resize_token_embeddings(len(tokenizer))
    
    # Training arguments - GPU optimized
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Start smaller to avoid OOM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # Effectively batch size of 16
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train model
    print(f"Training model on {device}...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error occurred: {e}")
            print("Falling back to CPU training...")
            # Move model to CPU and retry
            model = model.to('cpu')
            training_args.no_cuda = True
            training_args.fp16 = False
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )
            trainer.train()
        else:
            raise e
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save model components
    model_data = {
        'model': model,
        'tokenizer': tokenizer,
        'label_names': ['data_collection', 'data_sharing', 'data_storage', 'vague_language']
    }
    
    return model_data

def create_summarization_pipeline():
    """Create a summarization pipeline"""
    print("Creating summarization pipeline...")
    
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )
    
    return summarizer

def save_model(model_data, summarizer):
    """Save the trained model and components"""
    print("Saving model...")
    
    model_package = {
        'classifier': model_data,
        'summarizer': summarizer,
        'version': '1.0',
        'categories': ['data_collection', 'data_sharing', 'data_storage', 'vague_language']
    }
    
    with open('privacy_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("Model saved as privacy_model.pkl")

def main():
    """Main training pipeline"""
    print("Starting Privacy Policy Analysis Model Training with Real OPP-115 Data...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Download dataset if not already present
    if not os.path.exists("data/OPP-115"):
        download_success = download_opp115_dataset()
        if not download_success:
            print("Failed to download dataset. Exiting...")
            return
    else:
        print("Dataset directory already exists, skipping download.")
    
    # Load real OPP-115 data
    df = load_real_opp115_data()
    
    if df is None or len(df) == 0:
        print("Failed to load real OPP-115 data. Exiting...")
        return
    
    # Create and train classifier
    model_data = create_multi_label_classifier(df)
    
    # Create summarization pipeline
    summarizer = create_summarization_pipeline()
    
    # Save model
    save_model(model_data, summarizer)
    
    print("Training completed successfully!")
    print("Model saved as 'privacy_model.pkl'")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()