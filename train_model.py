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

class SimpleSummarizer:
    """Enhanced extractive summarization class for privacy policies"""
    
    def __init__(self):
        # Comprehensive privacy policy keywords organized by category
        self.data_collection_keywords = [
            'collect', 'gather', 'obtain', 'receive', 'acquire', 'capture',
            'personal information', 'personal data', 'user data', 'user information',
            'name', 'email', 'address', 'phone', 'location', 'browsing', 'ip address',
            'cookies', 'tracking', 'analytics', 'profile'
        ]
        
        self.data_sharing_keywords = [
            'share', 'disclose', 'provide', 'transfer', 'sell', 'distribute',
            'third party', 'third-party', 'partners', 'affiliates', 'vendors',
            'service providers', 'marketing', 'advertising', 'analytics companies'
        ]
        
        self.data_storage_keywords = [
            'store', 'retain', 'keep', 'maintain', 'preserve', 'hold',
            'servers', 'database', 'cloud', 'security', 'encryption', 'protect',
            'secure', 'safeguard', 'unauthorized access', 'data breach'
        ]
        
        self.user_rights_keywords = [
            'rights', 'access', 'delete', 'modify', 'correct', 'update',
            'opt-out', 'unsubscribe', 'withdraw consent', 'control',
            'choice', 'preferences', 'request', 'contact'
        ]
        
        self.policy_updates_keywords = [
            'update', 'change', 'modify', 'revise', 'amend', 'notify',
            'notification', 'effective date', 'changes to policy'
        ]
        
        # Category weights for balanced summary
        self.category_weights = {
            'collection': 1.5,  # High priority
            'sharing': 1.4,     # High priority  
            'storage': 1.2,     # Medium-high priority
            'rights': 1.3,      # Medium-high priority
            'updates': 1.0      # Medium priority
        }
    
    def _preprocess_text(self, text):
        """Better text preprocessing"""
        import re
        from collections import defaultdict
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Better sentence splitting that handles common abbreviations
        # Split on sentence endings but preserve abbreviations
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences, headers, and navigation text
            if (len(sentence) > 25 and 
                not sentence.isupper() and  # Skip all-caps headers
                not re.match(r'^(home|about|contact|privacy|terms)', sentence.lower()) and
                '.' in sentence):  # Must contain a period to be a complete sentence
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _categorize_sentence(self, sentence):
        """Categorize sentence by privacy policy section"""
        from collections import defaultdict
        
        sentence_lower = sentence.lower()
        scores = defaultdict(int)
        
        # Check for data collection indicators
        for keyword in self.data_collection_keywords:
            if keyword in sentence_lower:
                scores['collection'] += 1
        
        # Check for data sharing indicators  
        for keyword in self.data_sharing_keywords:
            if keyword in sentence_lower:
                scores['sharing'] += 1
                
        # Check for data storage indicators
        for keyword in self.data_storage_keywords:
            if keyword in sentence_lower:
                scores['storage'] += 1
                
        # Check for user rights indicators
        for keyword in self.user_rights_keywords:
            if keyword in sentence_lower:
                scores['rights'] += 1
                
        # Check for policy updates indicators
        for keyword in self.policy_updates_keywords:
            if keyword in sentence_lower:
                scores['updates'] += 1
        
        # Return the category with highest score, or None if no match
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _score_sentence(self, sentence, position_ratio):
        """Enhanced sentence scoring algorithm"""
        sentence_lower = sentence.lower()
        score = 0
        
        # Base score from sentence length (but cap it to avoid overly long sentences)
        length_score = min(len(sentence) * 0.1, 20)
        score += length_score
        
        # Position scoring - earlier sentences get slight boost
        position_score = (1.0 - position_ratio) * 10
        score += position_score
        
        # Keyword matching with category weights
        category = self._categorize_sentence(sentence)
        if category:
            category_weight = self.category_weights.get(category, 1.0)
            
            # Count keyword matches in this category
            keyword_matches = 0
            if category == 'collection':
                keyword_matches = sum(1 for kw in self.data_collection_keywords if kw in sentence_lower)
            elif category == 'sharing':
                keyword_matches = sum(1 for kw in self.data_sharing_keywords if kw in sentence_lower)
            elif category == 'storage':
                keyword_matches = sum(1 for kw in self.data_storage_keywords if kw in sentence_lower)
            elif category == 'rights':
                keyword_matches = sum(1 for kw in self.user_rights_keywords if kw in sentence_lower)
            elif category == 'updates':
                keyword_matches = sum(1 for kw in self.policy_updates_keywords if kw in sentence_lower)
            
            # Apply category weight and keyword bonus
            score += keyword_matches * 15 * category_weight
        
        # Bonus for specific important phrases
        important_phrases = [
            'personal information', 'third party', 'third-party', 'we collect',
            'we share', 'we store', 'we use', 'you can', 'your rights',
            'contact us', 'opt out', 'delete your', 'access your'
        ]
        
        phrase_bonus = sum(10 for phrase in important_phrases if phrase in sentence_lower)
        score += phrase_bonus
        
        # Penalty for generic/vague sentences
        vague_indicators = ['may', 'might', 'could', 'generally', 'typically', 'usually']
        vague_penalty = sum(5 for indicator in vague_indicators if indicator in sentence_lower)
        score -= vague_penalty
        
        return score
    
    def _select_diverse_sentences(self, scored_sentences, target_count=3):
        """Select sentences that cover different aspects of privacy policy"""
        if len(scored_sentences) <= target_count:
            return [sent[1] for sent in scored_sentences]
        
        selected = []
        categories_covered = set()
        
        # Sort by score first
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # First pass: Select top sentence from each category
        for score, sentence in scored_sentences:
            category = self._categorize_sentence(sentence)
            if category and category not in categories_covered:
                selected.append(sentence)
                categories_covered.add(category)
                if len(selected) >= target_count:
                    break
        
        # Second pass: Fill remaining slots with highest scoring sentences
        if len(selected) < target_count:
            for score, sentence in scored_sentences:
                if sentence not in selected:
                    selected.append(sentence)
                    if len(selected) >= target_count:
                        break
        
        return selected[:target_count]
    
    def __call__(self, text, max_length=250, min_length=50):
        """Enhanced extractive summarization"""
        if not text or len(text.strip()) < 50:
            return text if text else "No content to summarize."
        
        # Preprocess and extract sentences
        sentences = self._preprocess_text(text)
        
        if len(sentences) <= 2:
            # If very few sentences, return truncated text
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Score all sentences
        scored_sentences = []
        total_sentences = len(sentences)
        
        for i, sentence in enumerate(sentences):
            position_ratio = i / total_sentences if total_sentences > 1 else 0
            score = self._score_sentence(sentence, position_ratio)
            scored_sentences.append((score, sentence))
        
        # Select diverse, high-quality sentences - try for more coverage
        target_sentences = min(5, max(3, len(sentences) // 2))  # More adaptive count
        selected_sentences = self._select_diverse_sentences(scored_sentences, target_sentences)
        
        # Create summary
        summary = '. '.join(selected_sentences)
        
        # Ensure summary fits length constraints
        if len(summary) > max_length:
            # Truncate to max_length but try to end at a sentence boundary
            truncated = summary[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # If we can keep most of the content
                summary = truncated[:last_period + 1]
            else:
                summary = truncated + "..."
        
        # Ensure minimum length by adding more sentences if needed
        if len(summary) < min_length and len(selected_sentences) < len(sentences):
            remaining_sentences = [sent[1] for sent in scored_sentences 
                                 if sent[1] not in selected_sentences]
            for sentence in remaining_sentences:
                test_summary = summary + '. ' + sentence
                if len(test_summary) <= max_length:
                    summary = test_summary
                if len(summary) >= min_length:
                    break
        
        return summary if summary.strip() else "Unable to generate meaningful summary."

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

def generate_synthetic_privacy_data():
    """Generate synthetic privacy policy data for training when real data is unavailable"""
    print("Generating synthetic privacy policy data...")
    
    # Sample privacy policy text segments with different characteristics
    synthetic_data = [
        # Data collection examples
        {
            'text': "We collect personal information including your name, email address, phone number, and location data when you use our services to provide you with a personalized experience.",
            'labels': {'data_collection': 1, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 0}
        },
        {
            'text': "Our application automatically gathers device information, usage statistics, and browsing behavior to improve our service quality and user experience.",
            'labels': {'data_collection': 1, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 1}
        },
        {
            'text': "We obtain information about your preferences, interests, and demographic data through surveys and interaction with our platform.",
            'labels': {'data_collection': 1, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 0}
        },
        
        # Data sharing examples
        {
            'text': "We may share your personal information with third-party partners, affiliates, and service providers for business purposes and marketing activities.",
            'labels': {'data_collection': 0, 'data_sharing': 1, 'data_storage': 0, 'vague_language': 1}
        },
        {
            'text': "Your data will be disclosed to law enforcement agencies when required by law or to protect our legitimate business interests.",
            'labels': {'data_collection': 0, 'data_sharing': 1, 'data_storage': 0, 'vague_language': 1}
        },
        {
            'text': "We transfer information to our trusted partners and subsidiaries who help us operate our business and provide services to you.",
            'labels': {'data_collection': 0, 'data_sharing': 1, 'data_storage': 0, 'vague_language': 0}
        },
        
        # Data storage examples
        {
            'text': "We store your personal data on secure servers with encryption and implement appropriate technical safeguards to protect your information.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 1, 'vague_language': 0}
        },
        {
            'text': "Your information is retained for as long as necessary to fulfill the purposes outlined in this privacy policy and to comply with legal obligations.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 1, 'vague_language': 1}
        },
        {
            'text': "We maintain data security through industry-standard encryption, access controls, and regular security audits of our storage systems.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 1, 'vague_language': 0}
        },
        
        # Vague language examples
        {
            'text': "We may use your information for other purposes as we deem necessary or appropriate to enhance user experience and improve our services.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 1}
        },
        {
            'text': "From time to time, we might update our practices and procedures in our sole discretion to better serve our legitimate business interests.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 1}
        },
        {
            'text': "We reserve the right to modify these terms as needed for business purposes and regulatory compliance requirements.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 1}
        },
        
        # Clear, neutral examples
        {
            'text': "This privacy policy explains how we handle information and describes our commitment to protecting user privacy.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 0}
        },
        {
            'text': "You can contact our privacy team at privacy@company.com if you have questions about this policy or your rights.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 0}
        },
        {
            'text': "This document was last updated on January 1, 2024 and becomes effective immediately upon posting.",
            'labels': {'data_collection': 0, 'data_sharing': 0, 'data_storage': 0, 'vague_language': 0}
        },
    ]
    
    # Expand the dataset by creating variations
    expanded_data = []
    for item in synthetic_data:
        expanded_data.append(item)
        
        # Create variations with slight modifications
        for i in range(3):
            variation = item.copy()
            # Add some noise to text while preserving meaning
            text = variation['text']
            
            # Simple variations
            if 'information' in text:
                text = text.replace('information', 'data' if i == 0 else 'details' if i == 1 else 'information')
            if 'we' in text.lower():
                text = text.replace('We ', 'Our company ' if i == 0 else 'This organization ' if i == 1 else 'We ')
            
            variation['text'] = text
            expanded_data.append(variation)
    
    # Convert to DataFrame format
    texts = [item['text'] for item in expanded_data]
    df_data = {'text': texts}
    
    for label in ['data_collection', 'data_sharing', 'data_storage', 'vague_language']:
        df_data[label] = [item['labels'][label] for item in expanded_data]
    
    df = pd.DataFrame(df_data)
    print(f"Generated {len(df)} synthetic training samples")
    
    # Print label distribution
    for label in ['data_collection', 'data_sharing', 'data_storage', 'vague_language']:
        count = df[label].sum()
        print(f"{label}: {count} positive samples ({count/len(df)*100:.1f}%)")
    
    return df

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
    
    # Use TF-IDF Vectorizer for text features (offline-friendly approach)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    # Create text processing pipeline
    print("Using TF-IDF + Logistic Regression approach (offline-friendly)")
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
    except ValueError:
        print("Using simple split")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    # Create multi-output classifier
    classifier = MultiOutputClassifier(
        LogisticRegression(random_state=42, max_iter=1000)
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train the model
    print("Training TF-IDF + Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Create predictions for detailed evaluation
    y_pred = pipeline.predict(X_test)
    
    # Print classification report for each label
    label_names = ['data_collection', 'data_sharing', 'data_storage', 'vague_language']
    for i, label_name in enumerate(label_names):
        print(f"\n{label_name}:")
        print(classification_report(y_test[:, i], y_pred[:, i], zero_division=0))
    
    # Save model components - lightweight version
    model_data = {
        'pipeline': pipeline,
        'vectorizer': vectorizer,
        'classifier': classifier,
        'label_names': label_names,
        'model_type': 'sklearn_tfidf'
    }
    
    return model_data

def create_summarization_pipeline():
    """Create a simple extractive summarization function"""
    print("Creating simple extractive summarization...")
    return SimpleSummarizer()

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
            print("Failed to download dataset. Will use synthetic data instead...")
    else:
        print("Dataset directory already exists, skipping download.")
    
    # Load real OPP-115 data
    df = load_real_opp115_data()
    
    if df is None or len(df) == 0:
        print("Failed to load real OPP-115 data. Using synthetic data instead...")
        df = generate_synthetic_privacy_data()
        
        if df is None or len(df) == 0:
            print("Failed to generate synthetic data. Exiting...")
            return
    
    # Create and train classifier
    model_data = create_multi_label_classifier(df)
    
    # Create summarization pipeline
    summarizer = create_summarization_pipeline()
    
    # Save model
    save_model(model_data, summarizer)
    
    print("Training completed successfully!")
    print("Model saved as 'privacy_model.pkl'")
    print("You can now run the Streamlit app with: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()