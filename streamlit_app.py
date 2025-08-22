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
        st.error("Model file 'privacy_model.pkl' not found. Please run 'train_model.py' first!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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
        classifier = model_package['classifier']
        model = classifier['model']
        tokenizer = classifier['tokenizer']
        label_names = classifier['label_names']
        
        # Force model to CPU for inference
        model = model.to('cpu')
        model.eval()
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize - don't return tensors initially
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
    analysis_type = st.sidebar.radio(
        "Choose Input Method:",
        ["📄 Upload PDF", "🔗 Enter URL", "📝 Paste Text"]
    )
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Main content area
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
        
        if st.button("🔍 Analyze Privacy Policy", type="primary"):
            with st.spinner("Analyzing privacy policy..."):
                
                # Classification
                st.subheader("📋 Quick Summary")
                summary = summarize_text(text_content, model_package)
                st.markdown(f"""
                <div class="card">
                    <h4>📝 Policy Summary</h4>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
                
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