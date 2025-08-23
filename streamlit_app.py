#!/usr/bin/env python3
"""
Privacy Policy Analysis Streamlit App
Analyzes privacy policies using trained ML models
"""

import streamlit as st
import pickle
import torch
import numpy as np
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import re
import warnings
warnings.filterwarnings('ignore')

# Import the SimpleSummarizer class from train_model
try:
    from train_model import SimpleSummarizer
except ImportError:
    # Fallback definition if import fails
    class SimpleSummarizer:
        def __init__(self):
            self.keywords = ['data', 'information', 'collect', 'share', 'privacy', 'policy', 'personal']
        
        def __call__(self, text, max_length=150, min_length=50):
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= 3:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            scored_sentences = []
            for sentence in sentences[:10]:
                score = len(sentence)
                for keyword in self.keywords:
                    if keyword.lower() in sentence.lower():
                        score += 50
                scored_sentences.append((score, sentence))
            
            scored_sentences.sort(reverse=True)
            selected_sentences = [sent[1] for sent in scored_sentences[:3]]
            
            summary = '. '.join(selected_sentences)
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary

# Page configuration
st.set_page_config(
    page_title="Privacy Policy Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-card {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        border: 1px solid #ffeaa7;
    }
    
    .success-card {
        background-color: #d4edda;
        border-left-color: #28a745;
        border: 1px solid #c3e6cb;
    }
    
    .danger-card {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        border: 1px solid #f5c6cb;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained privacy policy analysis model"""
    try:
        with open('privacy_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def display_model_setup_instructions():
    """Display instructions for setting up the model"""
    st.error("🚨 Model not found!")
    
    st.markdown("""
    <div class="card alert-card">
        <h4>📝 Setup Required</h4>
        <p>The privacy policy analysis model hasn't been trained yet. Please follow these steps:</p>
        <ol>
            <li><strong>Install dependencies:</strong> <code>pip install -r requirements.txt</code></li>
            <li><strong>Train the model:</strong> <code>python train_model.py</code></li>
            <li><strong>Start the app:</strong> <code>streamlit run streamlit_app.py</code></li>
        </ol>
        <p><strong>Note:</strong> The training script will automatically use synthetic data if the OPP-115 dataset is not available.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Check for Model Again"):
        st.rerun()

def extract_text_from_url(url):
    """Extract text content from a privacy policy URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def preprocess_text(text, max_length=512):
    """Preprocess text for model input"""
    # Clean text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()
    
    # Truncate if too long (keeping in mind tokenizer limits)
    words = text.split()
    if len(words) > max_length:
        text = ' '.join(words[:max_length])
    
    return text

def classify_privacy_policy(text, model_package):
    """Classify privacy policy text using the trained model"""
    try:
        classifier_data = model_package['classifier']
        
        # Check if this is the new sklearn model or old transformer model
        if 'model_type' in classifier_data and classifier_data['model_type'] == 'sklearn_tfidf':
            # New sklearn-based model
            pipeline = classifier_data['pipeline']
            label_names = classifier_data['label_names']
            
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Predict probabilities
            probabilities = pipeline.predict_proba([processed_text])
            predictions = pipeline.predict([processed_text])[0]
            
            # Convert to results format
            results = {}
            for i, label in enumerate(label_names):
                # For sklearn, we get probability for class 1 (positive class)
                prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else 0.5
                results[label] = {
                    'probability': float(prob),
                    'predicted': bool(predictions[i])
                }
            
            return results
            
        else:
            # Original transformer model (fallback)
            model = classifier_data['model']
            tokenizer = classifier_data['tokenizer']
            label_names = classifier_data['label_names']
            
            # Force model to CPU for inference
            model = model.to('cpu')
            model.eval()
            
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Tokenize
            inputs = tokenizer(
                processed_text, 
                truncation=True, 
                padding=True, 
                max_length=256,
                return_tensors="pt"
            )
            
            # Move inputs to CPU
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.sigmoid(outputs.logits)
            
            # Convert to probabilities and create results
            probs = predictions.cpu().numpy()[0]
            results = {}
            
            for i, label in enumerate(label_names):
                results[label] = {
                    'probability': float(probs[i]),
                    'predicted': probs[i] > 0.5
                }
            
            return results
            
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None

def summarize_text(text, model_package):
    """Summarize privacy policy text"""
    try:
        summarizer = model_package['summarizer']
        
        # Check if this is a function (simple summarizer) or pipeline (transformer)
        if callable(summarizer):
            # Simple extractive summarizer
            return summarizer(text)
        else:
            # Original transformer summarizer
            # Limit text length for summarization
            words = text.split()
            if len(words) > 1024:
                text = ' '.join(words[:1024])
            
            # Generate summary - force CPU usage
            summary = summarizer(
                text, 
                max_length=150, 
                min_length=50, 
                do_sample=False,
                device=-1  # Force CPU
            )
            return summary[0]['summary_text']
            
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return "Unable to generate summary."

def analyze_vague_language(text):
    """Analyze text for vague or ambiguous language"""
    vague_indicators = [
        'may', 'might', 'could', 'possible', 'reasonable', 'appropriate',
        'necessary', 'discretion', 'from time to time', 'as needed',
        'business purposes', 'legitimate interests', 'improve services',
        'enhance user experience', 'other purposes', 'similar purposes'
    ]
    
    found_indicators = []
    text_lower = text.lower()
    
    for indicator in vague_indicators:
        if indicator in text_lower:
            # Find context around the vague term
            pattern = rf'.{{0,50}}{re.escape(indicator)}.{{0,50}}'
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                context = text[match.start():match.end()].strip()
                found_indicators.append({
                    'term': indicator,
                    'context': context
                })
                break  # Only store first occurrence per term
    
    return found_indicators

def display_step_by_step_analysis(text, model_package):
    """Display step-by-step NLP analysis with intermediate results"""
    st.subheader("🔍 Step-by-Step NLP Analysis")
    
    # Step 1: Text Preprocessing
    with st.expander("Step 1: Text Preprocessing", expanded=False):
        st.markdown("**Original Text (first 500 chars):**")
        st.text(text[:500] + "..." if len(text) > 500 else text)
        
        processed_text = preprocess_text(text)
        st.markdown("**After Preprocessing:**")
        st.text(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
        
        st.markdown(f"""
        **Changes Made:**
        - Original length: {len(text)} characters, {len(text.split())} words
        - Processed length: {len(processed_text)} characters, {len(processed_text.split())} words
        - Whitespace normalized: ✅
        - Text truncated: {'✅' if len(text.split()) > 512 else '❌ (not needed)'}
        """)
    
    # Step 2: Feature Extraction
    with st.expander("Step 2: TF-IDF Feature Extraction", expanded=False):
        classifier_data = model_package['classifier']
        if 'pipeline' in classifier_data:
            vectorizer = classifier_data['pipeline'].named_steps['tfidf']
            
            # Transform the text to see features
            features = vectorizer.transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get non-zero features
            feature_indices = features.nonzero()[1]
            feature_scores = features.data
            
            st.markdown(f"""
            **TF-IDF Vectorization Results:**
            - Total vocabulary size: {len(feature_names):,}
            - Features extracted from this text: {len(feature_indices):,}
            - Feature density: {len(feature_indices)/len(feature_names)*100:.2f}%
            """)
            
            # Show top features
            if len(feature_indices) > 0:
                top_features = sorted(zip(feature_indices, feature_scores), 
                                    key=lambda x: x[1], reverse=True)[:20]
                
                st.markdown("**Top 20 TF-IDF Features:**")
                for idx, score in top_features:
                    feature_name = feature_names[idx]
                    st.write(f"• {feature_name}: {score:.4f}")
    
    # Step 3: Classification
    with st.expander("Step 3: Multi-Label Classification", expanded=False):
        # Get probabilities
        classifier_data = model_package['classifier']
        pipeline = classifier_data['pipeline']
        probabilities = pipeline.predict_proba([processed_text])
        predictions = pipeline.predict([processed_text])[0]
        label_names = classifier_data['label_names']
        
        st.markdown("**Classification Process:**")
        st.markdown("Each category is classified independently using logistic regression.")
        
        for i, label in enumerate(label_names):
            prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else 0.5
            prediction = bool(predictions[i])
            
            st.markdown(f"""
            **{label.replace('_', ' ').title()}:**
            - Raw probability: {prob:.4f}
            - Binary prediction: {'Positive' if prediction else 'Negative'} (threshold: 0.5)
            - Confidence: {'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'}
            """)
    
    # Step 4: Summarization
    with st.expander("Step 4: Extractive Summarization", expanded=False):
        summarizer = model_package['summarizer']
        
        # Show how summarization works
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        st.markdown(f"**Sentence Analysis:**")
        st.markdown(f"- Total sentences: {len(sentences)}")
        st.markdown(f"- Sentences considered: {min(len(sentences), 10)}")
        
        if len(sentences) > 3:
            keywords = ['data', 'information', 'collect', 'share', 'privacy', 'policy', 'personal']
            
            scored_sentences = []
            for sentence in sentences[:10]:
                score = len(sentence)
                keyword_matches = 0
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        score += 50
                        keyword_matches += 1
                scored_sentences.append((score, sentence, keyword_matches))
            
            scored_sentences.sort(reverse=True)
            
            st.markdown("**Top 5 Sentences by Score:**")
            for i, (score, sentence, keywords_found) in enumerate(scored_sentences[:5]):
                st.markdown(f"""
                **Rank {i+1}** (Score: {score}, Keywords: {keywords_found})
                {sentence[:200]}{'...' if len(sentence) > 200 else ''}
                """)
        
        summary = summarizer(text)
        st.markdown(f"**Final Summary:**")
        st.markdown(f'"{summary}"')
    
    # Step 5: Vague Language Detection
    with st.expander("Step 5: Vague Language Detection", expanded=False):
        vague_indicators = analyze_vague_language(text)
        
        patterns = [
            'may', 'might', 'could', 'possible', 'reasonable', 'appropriate',
            'necessary', 'discretion', 'from time to time', 'as needed',
            'business purposes', 'legitimate interests', 'improve services',
            'enhance user experience', 'other purposes', 'similar purposes'
        ]
        
        st.markdown(f"**Pattern Matching Results:**")
        st.markdown(f"- Patterns searched: {len(patterns)}")
        st.markdown(f"- Vague terms found: {len(vague_indicators)}")
        
        if vague_indicators:
            st.markdown("**Detected Vague Terms:**")
            for indicator in vague_indicators[:10]:
                st.markdown(f"• **{indicator['term']}**: {indicator['context'][:100]}...")
        else:
            st.markdown("✅ No vague language patterns detected")

def display_classification_results(results):
    """Display classification results in a visually appealing way"""
    st.subheader("🔍 Policy Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data Collection
        prob = results['data_collection']['probability']
        predicted = results['data_collection']['predicted']
        color = "success" if not predicted else "alert"
        
        st.markdown(f"""
        <div class="card {color}-card">
            <h4>📊 Data Collection</h4>
            <p><strong>Probability:</strong> {prob:.2%}</p>
            <p><strong>Status:</strong> {"Detected" if predicted else "Not Detected"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Storage
        prob = results['data_storage']['probability']
        predicted = results['data_storage']['predicted']
        color = "success" if predicted else "alert"
        
        st.markdown(f"""
        <div class="card {color}-card">
            <h4>🏗️ Data Storage</h4>
            <p><strong>Probability:</strong> {prob:.2%}</p>
            <p><strong>Status:</strong> {"Addressed" if predicted else "Not Addressed"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Data Sharing
        prob = results['data_sharing']['probability']
        predicted = results['data_sharing']['predicted']
        color = "danger" if predicted else "success"
        
        st.markdown(f"""
        <div class="card {color}-card">
            <h4>🤝 Data Sharing</h4>
            <p><strong>Probability:</strong> {prob:.2%}</p>
            <p><strong>Status:</strong> {"Detected" if predicted else "Not Detected"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Vague Language
        prob = results['vague_language']['probability']
        predicted = results['vague_language']['predicted']
        color = "danger" if predicted else "success"
        
        st.markdown(f"""
        <div class="card {color}-card">
            <h4>⚠️ Vague Language</h4>
            <p><strong>Probability:</strong> {prob:.2%}</p>
            <p><strong>Status:</strong> {"Detected" if predicted else "Not Detected"}</p>
        </div>
        """, unsafe_allow_html=True)

def display_vague_language_analysis(vague_indicators):
    """Display vague language analysis"""
    if vague_indicators:
        st.subheader("🚨 Potentially Vague Language Detected")
        
        for indicator in vague_indicators[:5]:  # Show top 5
            st.markdown(f"""
            <div class="card danger-card">
                <h5>Term: "{indicator['term']}"</h5>
                <p><strong>Context:</strong> "{indicator['context']}"</p>
                <p><small>This language may be intentionally ambiguous and could allow for broad interpretation.</small></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card success-card">
            <h4>✅ Clear Language</h4>
            <p>No obviously vague or ambiguous language patterns detected.</p>
        </div>
        """, unsafe_allow_html=True)

def display_nlp_methods():
    """Display detailed NLP methodology and techniques"""
    st.markdown('<h1 class="main-header">🧠 NLP Methods & Techniques</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This page explains the Natural Language Processing and Machine Learning methods implemented in this project.
    All techniques work together to analyze privacy policies and extract meaningful insights.
    """)
    
    # Text Preprocessing
    st.subheader("1. 📝 Text Preprocessing Pipeline")
    st.markdown("""
    <div class="card">
        <h4>Text Cleaning & Normalization</h4>
        <ul>
            <li><strong>Whitespace Normalization:</strong> Multiple spaces/tabs converted to single spaces</li>
            <li><strong>Text Truncation:</strong> Limited to 512 words to fit model constraints</li>
            <li><strong>Character Cleaning:</strong> Removes excessive newlines and special characters</li>
        </ul>
        <h4>Why This Matters:</h4>
        <p>Clean, normalized text ensures consistent model performance and prevents tokenization issues.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Extraction
    st.subheader("2. 🔤 Feature Extraction: TF-IDF Vectorization")
    st.markdown("""
    <div class="card">
        <h4>TF-IDF (Term Frequency-Inverse Document Frequency)</h4>
        <ul>
            <li><strong>Vocabulary Size:</strong> 5,000 most important features</li>
            <li><strong>N-grams:</strong> Unigrams (1-word) and Bigrams (2-word phrases)</li>
            <li><strong>Stop Words:</strong> Common English words filtered out</li>
            <li><strong>Min/Max Document Frequency:</strong> Terms must appear in 1-95% of documents</li>
        </ul>
        <h4>TF-IDF Formula:</h4>
        <code>TF-IDF(t,d) = TF(t,d) × log(N/DF(t))</code>
        <p><small>Where TF = term frequency, DF = document frequency, N = total documents</small></p>
        <h4>Why TF-IDF:</h4>
        <p>Captures both local term importance (TF) and global rarity (IDF), perfect for legal text analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Classification Model
    st.subheader("3. 🎯 Multi-Label Classification")
    st.markdown("""
    <div class="card">
        <h4>Logistic Regression with Multi-Output Classification</h4>
        <ul>
            <li><strong>Algorithm:</strong> Logistic Regression (L2 regularization)</li>
            <li><strong>Multi-Label:</strong> Predicts 4 categories simultaneously</li>
            <li><strong>Categories:</strong> Data Collection, Data Sharing, Data Storage, Vague Language</li>
            <li><strong>Output:</strong> Probability scores (0-1) for each category</li>
        </ul>
        <h4>Sigmoid Function:</h4>
        <code>P(y=1) = 1 / (1 + e^(-z))</code>
        <p><small>Where z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ</small></p>
        <h4>Why Multi-Label:</h4>
        <p>Privacy policies often contain multiple types of practices simultaneously.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summarization
    st.subheader("4. 📄 Extractive Summarization")
    st.markdown("""
    <div class="card">
        <h4>Keyword-Based Sentence Scoring</h4>
        <ul>
            <li><strong>Method:</strong> Extractive (selects existing sentences)</li>
            <li><strong>Scoring:</strong> Sentence length + keyword presence</li>
            <li><strong>Keywords:</strong> 'data', 'information', 'collect', 'share', 'privacy', 'policy', 'personal'</li>
            <li><strong>Selection:</strong> Top 3 highest-scoring sentences</li>
        </ul>
        <h4>Scoring Formula:</h4>
        <code>Score = sentence_length + (keyword_matches × 50)</code>
        <h4>Why Extractive:</h4>
        <p>Preserves original legal language while highlighting key privacy practices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vague Language Detection
    st.subheader("5. ⚠️ Vague Language Detection")
    st.markdown("""
    <div class="card">
        <h4>Pattern Matching & Context Extraction</h4>
        <ul>
            <li><strong>Method:</strong> Rule-based pattern matching</li>
            <li><strong>Indicators:</strong> 'may', 'might', 'reasonable', 'discretion', 'business purposes'</li>
            <li><strong>Context:</strong> ±50 characters around each match</li>
            <li><strong>Regex:</strong> Case-insensitive matching with context windows</li>
        </ul>
        <h4>Why Pattern Matching:</h4>
        <p>Legal vagueness follows predictable linguistic patterns that can be systematically detected.</p>
    </div>
    """, unsafe_allow_html=True)

def display_model_architecture():
    """Display model architecture and technical details"""
    st.markdown('<h1 class="main-header">🏗️ Model Architecture</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This page shows the complete machine learning architecture and data flow of our privacy policy analyzer.
    """)
    
    # Load model to show details
    model_package = load_model()
    if model_package:
        st.subheader("📊 Current Model Information")
        
        classifier_data = model_package['classifier']
        
        # Model type and version
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card success-card">
                <h4>Model Type</h4>
                <p><strong>sklearn_tfidf</strong></p>
                <p>TF-IDF + Logistic Regression</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card success-card">
                <h4>Version</h4>
                <p><strong>{model_package.get('version', 'Unknown')}</strong></p>
                <p>Categories: {len(model_package.get('categories', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Pipeline components
        st.subheader("🔄 Processing Pipeline")
        
        if 'pipeline' in classifier_data:
            pipeline = classifier_data['pipeline']
            
            st.markdown("""
            <div class="card">
                <h4>Pipeline Steps</h4>
                <ol>
                    <li><strong>TF-IDF Vectorizer</strong>
                        <ul>
                            <li>Max Features: 5,000</li>
                            <li>N-gram Range: (1, 2)</li>
                            <li>Stop Words: English</li>
                            <li>Min DF: 1, Max DF: 0.95</li>
                        </ul>
                    </li>
                    <li><strong>Multi-Output Classifier</strong>
                        <ul>
                            <li>Base: Logistic Regression</li>
                            <li>Max Iterations: 1,000</li>
                            <li>Random State: 42</li>
                            <li>Regularization: L2</li>
                        </ul>
                    </li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature information
        st.subheader("🎯 Classification Categories")
        categories = model_package.get('categories', [])
        
        for i, category in enumerate(categories):
            color = "success" if i % 2 == 0 else "alert"
            st.markdown(f"""
            <div class="card {color}-card">
                <h4>{category.replace('_', ' ').title()}</h4>
                <p>Binary classification: Detects presence/absence of {category.replace('_', ' ')} practices</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data flow diagram
    st.subheader("📈 Data Flow Architecture")
    st.markdown("""
    <div class="card">
        <h4>Processing Flow</h4>
        <div style="font-family: monospace; background: #f0f0f0; padding: 1rem; border-radius: 5px;">
        Input Text<br>
        &nbsp;&nbsp;↓<br>
        Text Preprocessing<br>
        &nbsp;&nbsp;↓<br>
        TF-IDF Vectorization<br>
        &nbsp;&nbsp;↓<br>
        Multi-Label Classification<br>
        &nbsp;&nbsp;↓<br>
        Probability Calculation<br>
        &nbsp;&nbsp;↓<br>
        Results + Recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical specifications
    st.subheader("⚙️ Technical Specifications")
    st.markdown("""
    <div class="card">
        <h4>Performance Characteristics</h4>
        <ul>
            <li><strong>Inference Speed:</strong> ~100ms per document</li>
            <li><strong>Memory Usage:</strong> ~50MB model size</li>
            <li><strong>Input Limit:</strong> 512 words (tokenizer constraint)</li>
            <li><strong>Output Format:</strong> Probability scores (0.0-1.0)</li>
            <li><strong>Threshold:</strong> 0.5 for binary predictions</li>
        </ul>
        <h4>Dependencies</h4>
        <ul>
            <li><strong>scikit-learn:</strong> TF-IDF + Logistic Regression</li>
            <li><strong>numpy:</strong> Numerical computations</li>
            <li><strong>pandas:</strong> Data manipulation</li>
            <li><strong>pickle:</strong> Model serialization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Privacy Policy Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <h3>🛡️ About This Tool</h3>
        <p>This AI-powered tool analyzes privacy policies to help you understand:</p>
        <ul>
            <li>What data is collected</li>
            <li>How data is stored & secured</li>
            <li>If data is shared with third parties</li>
            <li>Potential loopholes or vague language</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation:",
        ["🔍 Analyze Policy", "🧠 NLP Methods", "🏗️ Model Architecture"]
    )
    
    if page == "🔍 Analyze Policy":
        analysis_type = st.sidebar.radio(
            "Choose Input Method:",
            ["📄 Upload PDF", "🔗 Enter URL", "📝 Paste Text"]
        )
    else:
        analysis_type = None
    
    # Load model
    model_package = load_model()
    if model_package is None and page == "🔍 Analyze Policy":
        display_model_setup_instructions()
        st.stop()
    elif model_package is not None and page == "🔍 Analyze Policy":
        st.sidebar.success("✅ Model loaded successfully!")
    
    # Handle different pages
    if page == "🧠 NLP Methods":
        display_nlp_methods()
        return
    elif page == "🏗️ Model Architecture":
        display_model_architecture()
        return
    
    # Main analysis page
    text_content = None
    
    if analysis_type == "📄 Upload PDF":
        st.subheader("📄 Upload Privacy Policy PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                text_content = extract_text_from_pdf(uploaded_file)
                
            if text_content:
                st.success(f"✅ Text extracted! ({len(text_content.split())} words)")
                
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted Text", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200)
    
    elif analysis_type == "🔗 Enter URL":
        st.subheader("🔗 Enter Privacy Policy URL")
        url = st.text_input("Privacy Policy URL", placeholder="https://example.com/privacy-policy")
        
        if st.button("Extract Text from URL") and url:
            with st.spinner("Extracting text from URL..."):
                text_content = extract_text_from_url(url)
            
            if text_content:
                st.success(f"✅ Text extracted! ({len(text_content.split())} words)")
                
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted Text", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200)
    
    elif analysis_type == "📝 Paste Text":
        st.subheader("📝 Paste Privacy Policy Text")
        text_content = st.text_area("Privacy Policy Text", height=200, placeholder="Paste the privacy policy text here...")
        
        if text_content:
            st.success(f"✅ Text ready for analysis! ({len(text_content.split())} words)")
    
    # Analysis section
    if text_content and len(text_content.strip()) > 50:
        st.markdown("---")
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            show_quick_results = st.button("🔍 Quick Analysis", type="primary")
        with col2:
            show_detailed_analysis = st.button("🧠 Detailed NLP Analysis", type="secondary")
        
        if show_quick_results or show_detailed_analysis:
            with st.spinner("Analyzing privacy policy..."):
                
                # Always show summary first
                st.subheader("📋 Quick Summary")
                summary = summarize_text(text_content, model_package)
                st.markdown(f"""
                <div class="card">
                    <h4>📝 Policy Summary</h4>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detailed analysis if requested
                if show_detailed_analysis:
                    st.markdown("---")
                    display_step_by_step_analysis(text_content, model_package)
                    st.markdown("---")
                
                # Classification results
                classification_results = classify_privacy_policy(text_content, model_package)
                
                if classification_results:
                    display_classification_results(classification_results)
                    
                    # Vague language analysis
                    st.markdown("---")
                    vague_indicators = analyze_vague_language(text_content)
                    display_vague_language_analysis(vague_indicators)
                    
                    # Key recommendations
                    st.markdown("---")
                    st.subheader("💡 Key Recommendations")
                    
                    recommendations = []
                    
                    if classification_results['data_collection']['predicted']:
                        recommendations.append("• Review what personal data is being collected and why")
                    
                    if classification_results['data_sharing']['predicted']:
                        recommendations.append("• Check if your data will be shared with third parties")
                    
                    if classification_results['vague_language']['predicted']:
                        recommendations.append("• Be cautious of ambiguous language that could be interpreted broadly")
                    
                    if not classification_results['data_storage']['predicted']:
                        recommendations.append("• Look for more specific information about data storage and security")
                    
                    if not recommendations:
                        recommendations.append("• This policy appears to be relatively clear and straightforward")
                    
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="card">
                            <p>{rec}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Classification failed. Please check the model and try again.")
    
    elif text_content and len(text_content.strip()) <= 50:
        st.warning("⚠️ Please provide more text for analysis (at least 50 characters)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using Streamlit and Transformers</p>
        <p><small>⚠️ This tool provides automated analysis and should not replace legal advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()