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
import pandas as pd
from collections import Counter
import time
import psutil
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Import our enhanced NLP concepts module
try:
    from nlp_concepts import NLPAnalyzer
except ImportError:
    st.error("NLP Concepts module not found. Please ensure nlp_concepts.py is in the same directory.")
    NLPAnalyzer = None

# Import the Privacy Assistant
try:
    from privacy_assistant import PrivacyAssistant
except ImportError:
    st.error("Privacy Assistant module not found. Please ensure privacy_assistant.py is in the same directory.")
    PrivacyAssistant = None

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
            # Enhanced extractive summarizer - use longer max_length for better coverage
            return summarizer(text, max_length=300, min_length=75)
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

def display_nlp_concepts_demo():
    """Display comprehensive NLP concepts demonstration"""
    st.markdown('<h1 class="main-header">🧠 NLP Concepts Demonstration</h1>', unsafe_allow_html=True)
    
    if NLPAnalyzer is None:
        st.error("NLP Analyzer not available. Please check the nlp_concepts.py file.")
        return
    
    analyzer = NLPAnalyzer()
    
    # Sidebar for concept selection
    st.sidebar.markdown("### Select NLP Concepts to Explore")
    concepts = {
        "🔤 Tokenization": "tokenization",
        "🏷️ Part-of-Speech Tagging": "pos_tagging",
        "🏛️ Named Entity Recognition": "ner",
        "😊 Sentiment Analysis": "sentiment",
        "📊 Text Statistics": "statistics",
        "🔑 Keyword Extraction": "keywords",
        "📈 N-gram Analysis": "ngrams",
        "🎯 Topic Modeling": "topics",
        "🔍 Privacy Pattern Analysis": "privacy_patterns",
        "🎓 Language Complexity": "complexity"
    }
    
    selected_concepts = []
    for display_name, concept_id in concepts.items():
        if st.sidebar.checkbox(display_name, value=(concept_id in ["tokenization", "statistics", "keywords"])):
            selected_concepts.append(concept_id)
    
    # Text input section
    st.subheader("📝 Enter Text for Analysis")
    demo_text = """
    We collect personal information including your name, email address, and location data when you use our services. 
    This information may be shared with third-party partners for business purposes. We store your data securely 
    using industry-standard encryption. You have the right to opt-out of data sharing at any time by contacting 
    our privacy team at privacy@company.com. We reserve the right to update this policy as needed for compliance purposes.
    """
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            value=demo_text,
            height=150,
            help="Enter any text to see various NLP concepts in action"
        )
    
    with col2:
        st.markdown("**Sample Texts:**")
        if st.button("🔐 Privacy Policy Sample"):
            st.experimental_rerun()
        if st.button("📰 News Article Sample"):
            text_input = """
            The new artificial intelligence system demonstrates remarkable capabilities in natural language processing. 
            Researchers at the university have developed algorithms that can understand context and generate human-like responses. 
            This breakthrough could revolutionize how we interact with technology in the future.
            """
        if st.button("📧 Email Sample"):
            text_input = """
            Dear valued customer, we are writing to inform you about important updates to our service. 
            These changes will improve your experience and provide better security features. 
            Please review the attached documentation for more details.
            """
    
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
        return
    
    # Display analysis results
    st.markdown("---")
    
    # Tokenization
    if "tokenization" in selected_concepts:
        st.subheader("🔤 Tokenization Analysis")
        with st.expander("What is Tokenization?", expanded=False):
            st.markdown("""
            **Tokenization** is the process of breaking down text into smaller units (tokens) such as words, sentences, or characters.
            It's the foundation of most NLP tasks.
            """)
        
        tokens = analyzer.basic_tokenization(text_input)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Words", tokens['word_count'])
        col2.metric("Sentences", tokens['sentence_count'])
        col3.metric("Characters", tokens['char_count'])
        col4.metric("Paragraphs", tokens['paragraph_count'])
        
        with st.expander("View Tokenization Details"):
            st.write("**First 20 Word Tokens:**")
            st.write(tokens['word_tokens'][:20])
            st.write("**Sentences:**")
            for i, sent in enumerate(tokens['sentences'][:5], 1):
                st.write(f"{i}. {sent}")
    
    # Part-of-Speech Tagging
    if "pos_tagging" in selected_concepts:
        st.subheader("🏷️ Part-of-Speech (POS) Tagging")
        with st.expander("What is POS Tagging?", expanded=False):
            st.markdown("""
            **POS Tagging** assigns grammatical categories (noun, verb, adjective, etc.) to each word in the text.
            This helps understand the grammatical structure and meaning.
            """)
        
        pos_tags = analyzer.pos_tagging_simple(text_input)
        
        # Create POS distribution
        pos_counts = Counter([tag for _, tag in pos_tags])
        pos_df = pd.DataFrame(list(pos_counts.items()), columns=['POS Tag', 'Count'])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.bar_chart(pos_df.set_index('POS Tag'))
        
        with col2:
            st.write("**Sample Tagged Words:**")
            for word, tag in pos_tags[:15]:
                color = {
                    'NOUN': '🔵', 'VERB': '🟢', 'ADJ': '🟡',
                    'MODAL': '🟠', 'PREP': '🟣'
                }.get(tag, '⚪')
                st.write(f"{color} **{word}** ({tag})")
    
    # Named Entity Recognition
    if "ner" in selected_concepts:
        st.subheader("🏛️ Named Entity Recognition (NER)")
        with st.expander("What is NER?", expanded=False):
            st.markdown("""
            **Named Entity Recognition** identifies and classifies named entities (people, organizations, locations, etc.) in text.
            This is crucial for information extraction and understanding.
            """)
        
        entities = analyzer.named_entity_recognition(text_input)
        
        has_entities = any(entities.values())
        if has_entities:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    st.write(f"**{entity_type.replace('_', ' ').title()}:**")
                    for entity in entity_list[:5]:  # Show first 5
                        st.write(f"• {entity}")
        else:
            st.info("No entities detected in this text.")
    
    # Sentiment Analysis
    if "sentiment" in selected_concepts:
        st.subheader("😊 Sentiment Analysis")
        with st.expander("What is Sentiment Analysis?", expanded=False):
            st.markdown("""
            **Sentiment Analysis** determines the emotional tone of text (positive, negative, or neutral).
            For privacy policies, this can indicate user-friendliness.
            """)
        
        sentiment = analyzer.sentiment_analysis_simple(text_input)
        
        col1, col2, col3 = st.columns(3)
        
        sentiment_colors = {
            'positive': '🟢',
            'negative': '🔴',
            'neutral': '🟡'
        }
        
        col1.metric(
            f"{sentiment_colors[sentiment['sentiment']]} Sentiment",
            sentiment['sentiment'].title()
        )
        col2.metric("Positive Words", sentiment['positive'])
        col3.metric("Negative Words", sentiment['negative'])
        
        # Sentiment score visualization
        score = sentiment['score']
        if score > 0:
            st.success(f"Sentiment Score: +{score:.3f} (Positive bias)")
        elif score < 0:
            st.error(f"Sentiment Score: {score:.3f} (Negative bias)")
        else:
            st.info(f"Sentiment Score: {score:.3f} (Neutral)")
    
    # Text Statistics
    if "statistics" in selected_concepts:
        st.subheader("📊 Comprehensive Text Statistics")
        with st.expander("What are Text Statistics?", expanded=False):
            st.markdown("""
            **Text Statistics** provide quantitative measures of text complexity, readability, and structure.
            These metrics help assess document quality and accessibility:
            
            - **Lexical Diversity**: Ratio of unique words to total words (higher = more varied vocabulary)
            - **Perplexity**: Measures text predictability (lower = more coherent/predictable text)
            - **Flesch Reading Ease**: Readability score (higher = easier to read)
            """)
        
        stats = analyzer.text_statistics(text_input)
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Lexical Diversity", f"{stats['lexical_diversity']:.3f}")
        col2.metric("Avg Words/Sentence", f"{stats['avg_words_per_sentence']:.1f}")
        col3.metric("Avg Chars/Word", f"{stats['avg_chars_per_word']:.1f}")
        col4.metric("Perplexity", f"{stats['perplexity']:.1f}")
        
        # Readability score
        flesch_score = stats['flesch_reading_ease']
        if flesch_score >= 90:
            readability = "Very Easy 😊"
            color = "success"
        elif flesch_score >= 80:
            readability = "Easy 🙂"
            color = "success"
        elif flesch_score >= 70:
            readability = "Fairly Easy 😐"
            color = "info"
        elif flesch_score >= 60:
            readability = "Standard 😕"
            color = "warning"
        elif flesch_score >= 50:
            readability = "Fairly Difficult 😟"
            color = "warning"
        else:
            readability = "Difficult 😰"
            color = "error"
        
        # Perplexity interpretation
        perplexity = stats['perplexity']
        if perplexity == 0:
            perplexity_level = "N/A"
            perplexity_color = "secondary"
        elif perplexity <= 50:
            perplexity_level = "Very Coherent 🎯"
            perplexity_color = "success"
        elif perplexity <= 100:
            perplexity_level = "Coherent 👍"
            perplexity_color = "success"
        elif perplexity <= 200:
            perplexity_level = "Moderately Coherent 🤔"
            perplexity_color = "info"
        elif perplexity <= 500:
            perplexity_level = "Less Coherent 😐"
            perplexity_color = "warning"
        else:
            perplexity_level = "Incoherent 😕"
            perplexity_color = "error"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card {color}-card">
                <h4>📖 Readability Score</h4>
                <p><strong>Flesch Reading Ease:</strong> {flesch_score:.1f}</p>
                <p><strong>Level:</strong> {readability}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card {perplexity_color}-card">
                <h4>🎲 Text Perplexity</h4>
                <p><strong>Perplexity Score:</strong> {perplexity:.1f}</p>
                <p><strong>Level:</strong> {perplexity_level}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Keyword Extraction
    if "keywords" in selected_concepts:
        st.subheader("🔑 Keyword Extraction")
        with st.expander("What is Keyword Extraction?", expanded=False):
            st.markdown("""
            **Keyword Extraction** identifies the most important words or phrases in text using TF-IDF or frequency analysis.
            This helps understand the main topics and themes.
            """)
        
        keywords = analyzer.keyword_extraction_tfidf(text_input, top_k=15)
        
        if keywords:
            # Display keywords as a bar chart using Streamlit
            df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
            st.bar_chart(df_keywords.set_index('Keyword')['Score'])
            
            # Display keywords in columns
            col1, col2 = st.columns(2)
            mid = len(keywords) // 2
            
            with col1:
                st.write("**Top Keywords:**")
                for word, score in keywords[:mid]:
                    st.write(f"• {word}: {score:.3f}")
            
            with col2:
                st.write("**More Keywords:**")
                for word, score in keywords[mid:]:
                    st.write(f"• {word}: {score:.3f}")
    
    # N-gram Analysis
    if "ngrams" in selected_concepts:
        st.subheader("📈 N-gram Analysis")
        with st.expander("What are N-grams?", expanded=False):
            st.markdown("""
            **N-grams** are sequences of N consecutive words. They help identify common phrases and patterns in text.
            - Bigrams (2-grams): pairs of words
            - Trigrams (3-grams): sequences of three words
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bigrams (2-word phrases):**")
            bigrams = analyzer.ngram_analysis(text_input, n=2, top_k=10)
            for ngram, count in bigrams[:8]:
                st.write(f"• {ngram}: {count}")
        
        with col2:
            st.write("**Trigrams (3-word phrases):**")
            trigrams = analyzer.ngram_analysis(text_input, n=3, top_k=10)
            for ngram, count in trigrams[:8]:
                st.write(f"• {ngram}: {count}")
    
    # Topic Modeling
    if "topics" in selected_concepts:
        st.subheader("🎯 Topic Modeling")
        with st.expander("What is Topic Modeling?", expanded=False):
            st.markdown("""
            **Topic Modeling** automatically discovers hidden topics in text using algorithms like LDA (Latent Dirichlet Allocation).
            Each topic is represented by a set of related words.
            """)
        
        topics = analyzer.topic_modeling_simple(text_input, n_topics=3)
        
        if topics:
            for topic in topics:
                st.write(f"**Topic {topic['topic_id'] + 1}:**")
                topic_words = ", ".join(topic['words'])
                st.write(f"• Key words: {topic_words}")
        else:
            st.info("Not enough text for topic modeling. Try a longer document.")
    
    # Privacy Pattern Analysis
    if "privacy_patterns" in selected_concepts:
        st.subheader("🔍 Privacy-Specific Pattern Analysis")
        with st.expander("What is Privacy Pattern Analysis?", expanded=False):
            st.markdown("""
            **Privacy Pattern Analysis** looks for specific patterns related to data privacy, such as:
            - Data collection language
            - Data sharing indicators
            - Vague or ambiguous terms
            - User rights mentions
            """)
        
        patterns = analyzer.analyze_privacy_patterns(text_input)
        
        for category, data in patterns.items():
            if data['terms']:
                category_name = category.replace('_', ' ').title()
                st.write(f"**{category_name}:** {data['total_count']} occurrences")
                
                terms_text = ", ".join([f"{term['term']} ({term['count']})" for term in data['terms'][:5]])
                st.write(f"• {terms_text}")
    
    # Language Complexity
    if "complexity" in selected_concepts:
        st.subheader("🎓 Language Complexity Analysis")
        with st.expander("What is Language Complexity Analysis?", expanded=False):
            st.markdown("""
            **Language Complexity Analysis** measures how difficult the text is to understand by analyzing:
            - Average word length
            - Complex words (long or multi-syllabic)
            - Technical jargon usage
            """)
        
        complexity = analyzer.language_complexity_analysis(text_input)
        
        if complexity:
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Avg Word Length", f"{complexity['avg_word_length']:.1f} chars")
            col2.metric("Complex Word Ratio", f"{complexity['complex_word_ratio']:.1%}")
            col3.metric("Jargon Ratio", f"{complexity['jargon_ratio']:.1%}")
            
            if complexity['complex_words']:
                st.write("**Sample Complex Words:**")
                st.write(", ".join(complexity['complex_words']))
            
            if complexity['jargon_terms']:
                st.write("**Technical/Legal Jargon Found:**")
                st.write(", ".join(complexity['jargon_terms']))

def display_text_comparison_tool():
    """Display text comparison and similarity analysis tool"""
    st.markdown('<h1 class="main-header">📊 Text Comparison & Similarity Analysis</h1>', unsafe_allow_html=True)
    
    if NLPAnalyzer is None:
        st.error("NLP Analyzer not available. Please check the nlp_concepts.py file.")
        return
    
    analyzer = NLPAnalyzer()
    
    st.markdown("""
    Compare two texts to analyze their similarity and differences. This is useful for:
    - Comparing different versions of privacy policies
    - Analyzing similar documents from different companies
    - Understanding content overlap and uniqueness
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Text 1")
        text1 = st.text_area(
            "Enter first text:",
            height=200,
            placeholder="Paste the first text here..."
        )
    
    with col2:
        st.subheader("📄 Text 2")
        text2 = st.text_area(
            "Enter second text:",
            height=200,
            placeholder="Paste the second text here..."
        )
    
    if text1.strip() and text2.strip():
        st.markdown("---")
        
        # Calculate similarity
        similarity = analyzer.text_similarity(text1, text2)
        
        # Display similarity score
        st.subheader("🔍 Similarity Analysis")
        
        similarity_percentage = similarity * 100
        
        if similarity_percentage >= 80:
            similarity_level = "Very High 🟢"
            color = "success"
        elif similarity_percentage >= 60:
            similarity_level = "High 🟡"
            color = "warning"
        elif similarity_percentage >= 40:
            similarity_level = "Moderate 🟠"
            color = "warning"
        elif similarity_percentage >= 20:
            similarity_level = "Low 🔴"
            color = "error"
        else:
            similarity_level = "Very Low ⚫"
            color = "error"
        
        st.markdown(f"""
        <div class="card {color}-card">
            <h4>📈 Similarity Score</h4>
            <p><strong>Score:</strong> {similarity_percentage:.1f}%</p>
            <p><strong>Level:</strong> {similarity_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparative analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Text 1 Analysis")
            stats1 = analyzer.text_statistics(text1)
            keywords1 = analyzer.keyword_extraction_tfidf(text1, top_k=10)
            
            st.metric("Word Count", stats1['word_count'])
            st.metric("Unique Words", stats1['unique_words'])
            st.metric("Readability", f"{stats1['flesch_reading_ease']:.0f}")
            
            st.write("**Top Keywords:**")
            for word, score in keywords1[:5]:
                st.write(f"• {word}")
        
        with col2:
            st.subheader("📊 Text 2 Analysis")
            stats2 = analyzer.text_statistics(text2)
            keywords2 = analyzer.keyword_extraction_tfidf(text2, top_k=10)
            
            st.metric("Word Count", stats2['word_count'])
            st.metric("Unique Words", stats2['unique_words'])
            st.metric("Readability", f"{stats2['flesch_reading_ease']:.0f}")
            
            st.write("**Top Keywords:**")
            for word, score in keywords2[:5]:
                st.write(f"• {word}")
        
        # Common and unique elements
        st.subheader("🔍 Content Analysis")
        
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        common_words = words1.intersection(words2)
        unique_to_1 = words1 - words2
        unique_to_2 = words2 - words1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Common Words:**")
            st.write(f"{len(common_words)} words")
            if common_words:
                sample_common = list(common_words)[:10]
                st.write(", ".join(sample_common))
        
        with col2:
            st.write("**Unique to Text 1:**")
            st.write(f"{len(unique_to_1)} words")
            if unique_to_1:
                sample_unique1 = list(unique_to_1)[:10]
                st.write(", ".join(sample_unique1))
        
        with col3:
            st.write("**Unique to Text 2:**")
            st.write(f"{len(unique_to_2)} words")
            if unique_to_2:
                sample_unique2 = list(unique_to_2)[:10]
                st.write(", ".join(sample_unique2))
    
    else:
        st.info("Please enter text in both fields to compare.")

def display_stats_for_nerds():
    """Display detailed stats about NLP processes and comprehensive NLP concepts demonstration"""
    st.markdown('<h1 class="main-header">🤓 Stats for Nerds</h1>', unsafe_allow_html=True)
    
    if NLPAnalyzer is None:
        st.error("NLP Analyzer not available. Please check the nlp_concepts.py file.")
        return
    
    st.markdown("""
    <div class="card">
        <h3>🔍 Advanced NLP Analysis & Performance Monitoring</h3>
        <p>This section provides both interactive NLP concept demonstrations and detailed performance monitoring of our NLP pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = NLPAnalyzer()
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🧠 NLP Concepts Explorer", "⚡ Performance Monitor"])
    
    with tab1:
        # NLP Concepts demonstration section (merged from display_nlp_concepts_demo)
        st.subheader("🧠 Interactive NLP Concepts Demonstration")
        
        # Sidebar for concept selection
        st.sidebar.markdown("### Select NLP Concepts to Explore")
        concepts = {
            "🔤 Tokenization": "tokenization",
            "🏷️ Part-of-Speech Tagging": "pos_tagging", 
            "🏛️ Named Entity Recognition": "ner",
            "😊 Sentiment Analysis": "sentiment",
            "📊 Text Statistics": "statistics",
            "🔑 Keyword Extraction": "keywords",
            "📈 N-gram Analysis": "ngrams",
            "🎯 Topic Modeling": "topics",
            "🔍 Privacy Pattern Analysis": "privacy_patterns",
            "🎓 Language Complexity": "complexity"
        }
        
        selected_concepts = []
        for display_name, concept_id in concepts.items():
            if st.sidebar.checkbox(display_name, value=(concept_id in ["tokenization", "statistics", "keywords"])):
                selected_concepts.append(concept_id)
        
        # Text input section
        st.subheader("📝 Enter Text for Analysis")
        demo_text = """
        We collect personal information including your name, email address, and location data when you use our services. 
        This information may be shared with third-party partners for business purposes. We store your data securely 
        using industry-standard encryption. You have the right to opt-out of data sharing at any time by contacting 
        our privacy team at privacy@company.com. We reserve the right to update this policy as needed for compliance purposes.
        """
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter text to analyze:",
                value=demo_text,
                height=150,
                help="Enter any text to see various NLP concepts in action"
            )
        
        with col2:
            st.markdown("**Sample Texts:**")
            if st.button("🔐 Privacy Policy Sample"):
                st.experimental_rerun()
            if st.button("📰 News Article Sample"):
                text_input = """
                The new artificial intelligence system demonstrates remarkable capabilities in natural language processing. 
                Researchers at the university have developed algorithms that can understand context and generate human-like responses. 
                This breakthrough could revolutionize how we interact with technology in the future.
                """
            if st.button("📧 Email Sample"):
                text_input = """
                Dear valued customer, we are writing to inform you about important updates to our service. 
                These changes will improve your experience and provide better security features. 
                Please review the attached documentation for more details.
                """
        
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
            
        # Display analysis results
        st.markdown("---")
        
        # Tokenization
        if "tokenization" in selected_concepts:
            st.subheader("🔤 Tokenization Analysis")
            with st.expander("What is Tokenization?", expanded=False):
                st.markdown("""
                **Tokenization** is the process of breaking down text into smaller units (tokens) such as words, sentences, or characters.
                It's the foundation of most NLP tasks.
                """)
            
            tokens = analyzer.basic_tokenization(text_input)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Words", tokens['word_count'])
            col2.metric("Sentences", tokens['sentence_count'])
            col3.metric("Characters", tokens['char_count'])
            col4.metric("Paragraphs", tokens['paragraph_count'])
            
            with st.expander("View Tokenization Details"):
                st.write("**First 20 Word Tokens:**")
                st.write(tokens['word_tokens'][:20])
                st.write("**Sentences:**")
                for i, sent in enumerate(tokens['sentences'][:5], 1):
                    st.write(f"{i}. {sent}")
        
        # Part-of-Speech Tagging
        if "pos_tagging" in selected_concepts:
            st.subheader("🏷️ Part-of-Speech (POS) Tagging")
            with st.expander("What is POS Tagging?", expanded=False):
                st.markdown("""
                **POS Tagging** assigns grammatical categories (noun, verb, adjective, etc.) to each word in the text.
                This helps understand the grammatical structure and meaning.
                """)
            
            pos_tags = analyzer.pos_tagging_simple(text_input)
            
            # Create POS distribution
            pos_counts = Counter([tag for _, tag in pos_tags])
            pos_df = pd.DataFrame(list(pos_counts.items()), columns=['POS Tag', 'Count'])
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.bar_chart(pos_df.set_index('POS Tag'))
            
            with col2:
                st.write("**Sample Tagged Words:**")
                for word, tag in pos_tags[:15]:
                    color = {
                        'NOUN': '🔵', 'VERB': '🟢', 'ADJ': '🟡',
                        'MODAL': '🟠', 'PREP': '🟣'
                    }.get(tag, '⚪')
                    st.write(f"{color} **{word}** ({tag})")
        
        # Named Entity Recognition
        if "ner" in selected_concepts:
            st.subheader("🏛️ Named Entity Recognition (NER)")
            with st.expander("What is NER?", expanded=False):
                st.markdown("""
                **Named Entity Recognition** identifies and classifies named entities (people, organizations, locations, etc.) in text.
                This is crucial for information extraction and understanding.
                """)
            
            entities = analyzer.named_entity_recognition(text_input)
            
            has_entities = any(entities.values())
            if has_entities:
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        st.write(f"**{entity_type.replace('_', ' ').title()}:**")
                        for entity in entity_list[:5]:  # Show first 5
                            st.write(f"• {entity}")
            else:
                st.info("No entities detected in this text.")
        
        # Sentiment Analysis
        if "sentiment" in selected_concepts:
            st.subheader("😊 Sentiment Analysis")
            with st.expander("What is Sentiment Analysis?", expanded=False):
                st.markdown("""
                **Sentiment Analysis** determines the emotional tone of text (positive, negative, or neutral).
                For privacy policies, this can indicate user-friendliness.
                """)
            
            sentiment = analyzer.sentiment_analysis_simple(text_input)
            
            sentiment_colors = {
                'positive': '🟢',
                'negative': '🔴',
                'neutral': '🟡'
            }
            
            col1, col2, col3 = st.columns(3)
            col1.metric(
                f"{sentiment_colors[sentiment['sentiment']]} Sentiment",
                sentiment['sentiment'].title()
            )
            col2.metric("Positive Words", sentiment['positive'])
            col3.metric("Negative Words", sentiment['negative'])
            
            # Sentiment score visualization
            score = sentiment['score']
            if score > 0:
                st.success(f"Sentiment Score: +{score:.3f} (Positive bias)")
            elif score < 0:
                st.error(f"Sentiment Score: {score:.3f} (Negative bias)")
            else:
                st.info(f"Sentiment Score: {score:.3f} (Neutral)")
        
        # Text Statistics
        if "statistics" in selected_concepts:
            st.subheader("📊 Comprehensive Text Statistics")
            with st.expander("What are Text Statistics?", expanded=False):
                st.markdown("""
                **Text Statistics** provide quantitative measures of text complexity, readability, and structure.
                These metrics help assess document quality and accessibility:
                
                - **Lexical Diversity**: Ratio of unique words to total words (higher = more varied vocabulary)
                - **Perplexity**: Measures text predictability (lower = more coherent/predictable text)
                - **Flesch Reading Ease**: Readability score (higher = easier to read)
                """)
            
            stats = analyzer.text_statistics(text_input)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Lexical Diversity", f"{stats['lexical_diversity']:.3f}")
            col2.metric("Avg Words/Sentence", f"{stats['avg_words_per_sentence']:.1f}")
            col3.metric("Avg Chars/Word", f"{stats['avg_chars_per_word']:.1f}")
            col4.metric("Perplexity", f"{stats['perplexity']:.1f}")
            
            # Readability score
            flesch_score = stats['flesch_reading_ease']
            if flesch_score >= 90:
                readability = "Very Easy 😊"
                color = "success"
            elif flesch_score >= 80:
                readability = "Easy 🙂"
                color = "success"
            elif flesch_score >= 70:
                readability = "Fairly Easy 😐"
                color = "info"
            elif flesch_score >= 60:
                readability = "Standard 😕"
                color = "warning"
            elif flesch_score >= 50:
                readability = "Fairly Difficult 😟"
                color = "warning"
            else:
                readability = "Difficult 😰"
                color = "error"
            
            # Perplexity interpretation
            perplexity = stats['perplexity']
            if perplexity == 0:
                perplexity_level = "N/A"
                perplexity_color = "secondary"
            elif perplexity <= 50:
                perplexity_level = "Very Coherent 🎯"
                perplexity_color = "success"
            elif perplexity <= 100:
                perplexity_level = "Coherent 👍"
                perplexity_color = "success"
            elif perplexity <= 200:
                perplexity_level = "Moderately Coherent 🤔"
                perplexity_color = "info"
            elif perplexity <= 500:
                perplexity_level = "Less Coherent 😐"
                perplexity_color = "warning"
            else:
                perplexity_level = "Incoherent 😕"
                perplexity_color = "error"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="card {color}-card">
                    <h4>📖 Readability Score</h4>
                    <p><strong>Flesch Reading Ease:</strong> {flesch_score:.1f}</p>
                    <p><strong>Level:</strong> {readability}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card {perplexity_color}-card">
                    <h4>🎲 Text Perplexity</h4>
                    <p><strong>Perplexity Score:</strong> {perplexity:.1f}</p>
                    <p><strong>Level:</strong> {perplexity_level}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Keyword Extraction
        if "keywords" in selected_concepts:
            st.subheader("🔑 Keyword Extraction")
            with st.expander("What is Keyword Extraction?", expanded=False):
                st.markdown("""
                **Keyword Extraction** identifies the most important words or phrases in text using TF-IDF or frequency analysis.
                This helps understand the main topics and themes.
                """)
            
            keywords = analyzer.keyword_extraction_tfidf(text_input, top_k=15)
            
            if keywords:
                # Display keywords as a bar chart using Streamlit
                df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
                st.bar_chart(df_keywords.set_index('Keyword')['Score'])
                
                # Display keywords in columns
                col1, col2 = st.columns(2)
                mid = len(keywords) // 2
                
                with col1:
                    st.write("**Top Keywords:**")
                    for word, score in keywords[:mid]:
                        st.write(f"• {word}: {score:.3f}")
                
                with col2:
                    st.write("**More Keywords:**")
                    for word, score in keywords[mid:]:
                        st.write(f"• {word}: {score:.3f}")
        
        # N-gram Analysis
        if "ngrams" in selected_concepts:
            st.subheader("📈 N-gram Analysis")
            with st.expander("What are N-grams?", expanded=False):
                st.markdown("""
                **N-grams** are sequences of N consecutive words. They help identify common phrases and patterns in text.
                - Bigrams (2-grams): pairs of words
                - Trigrams (3-grams): sequences of three words
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Bigrams (2-word phrases):**")
                bigrams = analyzer.ngram_analysis(text_input, n=2, top_k=10)
                for ngram, count in bigrams[:8]:
                    st.write(f"• {ngram}: {count}")
            
            with col2:
                st.write("**Trigrams (3-word phrases):**")
                trigrams = analyzer.ngram_analysis(text_input, n=3, top_k=10)
                for ngram, count in trigrams[:8]:
                    st.write(f"• {ngram}: {count}")
        
        # Topic Modeling
        if "topics" in selected_concepts:
            st.subheader("🎯 Topic Modeling")
            with st.expander("What is Topic Modeling?", expanded=False):
                st.markdown("""
                **Topic Modeling** automatically discovers hidden topics in text using algorithms like LDA (Latent Dirichlet Allocation).
                Each topic is represented by a set of related words.
                """)
            
            topics = analyzer.topic_modeling_simple(text_input, n_topics=3)
            
            if topics:
                for topic in topics:
                    st.write(f"**Topic {topic['topic_id'] + 1}:**")
                    topic_words = ", ".join(topic['words'])
                    st.write(f"• Key words: {topic_words}")
            else:
                st.info("Not enough text for topic modeling. Try a longer document.")
        
        # Privacy Pattern Analysis
        if "privacy_patterns" in selected_concepts:
            st.subheader("🔍 Privacy-Specific Pattern Analysis")
            with st.expander("What is Privacy Pattern Analysis?", expanded=False):
                st.markdown("""
                **Privacy Pattern Analysis** looks for specific patterns related to data privacy, such as:
                - Data collection language
                - Data sharing indicators
                - Vague or ambiguous terms
                - User rights mentions
                """)
            
            patterns = analyzer.analyze_privacy_patterns(text_input)
            
            for category, data in patterns.items():
                if data['terms']:
                    category_name = category.replace('_', ' ').title()
                    st.write(f"**{category_name}:** {data['total_count']} occurrences")
                    
                    terms_text = ", ".join([f"{term['term']} ({term['count']})" for term in data['terms'][:5]])
                    st.write(f"• {terms_text}")
        
        # Language Complexity
        if "complexity" in selected_concepts:
            st.subheader("🎓 Language Complexity Analysis")
            with st.expander("What is Language Complexity Analysis?", expanded=False):
                st.markdown("""
                **Language Complexity Analysis** measures how difficult the text is to understand by analyzing:
                - Average word length
                - Complex words (long or multi-syllabic)
                - Technical jargon usage
                """)
            
            complexity = analyzer.language_complexity_analysis(text_input)
            
            if complexity:
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Avg Word Length", f"{complexity['avg_word_length']:.1f} chars")
                col2.metric("Complex Word Ratio", f"{complexity['complex_word_ratio']:.1%}")
                col3.metric("Jargon Ratio", f"{complexity['jargon_ratio']:.1%}")
                
                if complexity['complex_words']:
                    st.write("**Sample Complex Words:**")
                    st.write(", ".join(complexity['complex_words']))
                
                if complexity['jargon_terms']:
                    st.write("**Technical/Legal Jargon Found:**")
                    st.write(", ".join(complexity['jargon_terms']))
    
    with tab2:
        # Performance monitoring section (original stats for nerds content)
        st.subheader("⚡ Real-time Performance Monitoring")
        
        # Text input section
        st.subheader("📝 Enter Text for Deep Analysis")
        demo_text_perf = """
        We collect personal information including your name, email address, and location data when you use our services. 
        This information may be shared with third-party partners for business purposes. We store your data securely 
        using industry-standard encryption. You have the right to opt-out of data sharing at any time by contacting 
        our privacy team at privacy@company.com. We reserve the right to update this policy as needed for compliance purposes.
        """
        
        text_input_perf = st.text_area(
            "Enter text to analyze:",
            value=demo_text_perf,
            height=150,
            help="Enter any text to see detailed processing statistics",
            key="perf_text_input"
        )
        
        if not text_input_perf.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Real-time processing with metrics
        st.markdown("---")
        st.subheader("⚡ Real-time Processing Pipeline")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Initialize metrics containers
        with col1:
            total_time_metric = st.empty()
        with col2:
            memory_metric = st.empty()
        with col3:
            steps_metric = st.empty()
        with col4:
            tokens_metric = st.empty()
        
        # Processing pipeline with timing
        pipeline_steps = []
        total_start_time = time.time()
        
        # Step 1: Tokenization
        step_start = time.time()
        tokens_perf = analyzer.basic_tokenization(text_input_perf)
        step_time = time.time() - step_start
        pipeline_steps.append({
            'step': 'Tokenization',
            'time_ms': step_time * 1000,
            'output_size': len(tokens_perf.get('word_tokens', [])),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        # Step 2: POS Tagging
        step_start = time.time()
        pos_tags_perf = analyzer.pos_tagging_simple(text_input_perf)
        step_time = time.time() - step_start
        pipeline_steps.append({
            'step': 'POS Tagging',
            'time_ms': step_time * 1000,
            'output_size': len(pos_tags_perf),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        # Step 3: Named Entity Recognition
        step_start = time.time()
        entities_perf = analyzer.named_entity_recognition(text_input_perf)
        step_time = time.time() - step_start
        entity_count = sum(len(ent_list) for ent_list in entities_perf.values())
        pipeline_steps.append({
            'step': 'Named Entity Recognition',
            'time_ms': step_time * 1000,
            'output_size': entity_count,
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        # Step 4: Sentiment Analysis
        step_start = time.time()
        sentiment_perf = analyzer.sentiment_analysis_simple(text_input_perf)
        step_time = time.time() - step_start
        pipeline_steps.append({
            'step': 'Sentiment Analysis',
            'time_ms': step_time * 1000,
            'output_size': 1,  # One sentiment result
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        # Step 5: Text Statistics
        step_start = time.time()
        stats_perf = analyzer.text_statistics(text_input_perf)
        step_time = time.time() - step_start
        pipeline_steps.append({
            'step': 'Text Statistics',
            'time_ms': step_time * 1000,
            'output_size': len(stats_perf),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        # Step 6: Keyword Extraction
        step_start = time.time()
        keywords_perf = analyzer.keyword_extraction_tfidf(text_input_perf, top_k=15)
        step_time = time.time() - step_start
        pipeline_steps.append({
            'step': 'Keyword Extraction',
            'time_ms': step_time * 1000,
            'output_size': len(keywords_perf),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        total_time = time.time() - total_start_time
        
        # Update metrics
        total_time_metric.metric("Total Processing Time", f"{total_time*1000:.1f} ms")
        memory_metric.metric("Memory Usage", f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        steps_metric.metric("Pipeline Steps", len(pipeline_steps))
        tokens_metric.metric("Total Tokens", len(tokens_perf.get('word_tokens', [])))
        
        # Create pipeline visualization
        st.subheader("🔄 Processing Pipeline Visualization")
        
        # Create DataFrame for pipeline steps
        df_pipeline = pd.DataFrame(pipeline_steps)
        
        # Create timing chart
        fig_timing = px.bar(
            df_pipeline, 
            x='step', 
            y='time_ms',
            title="Processing Time by Pipeline Step",
            labels={'time_ms': 'Time (milliseconds)', 'step': 'Pipeline Step'},
            color='time_ms',
            color_continuous_scale='viridis'
        )
        fig_timing.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_timing, use_container_width=True)
        
        # Memory usage chart
        fig_memory = px.line(
            df_pipeline,
            x='step',
            y='memory_mb',
            title="Memory Usage Throughout Pipeline",
            labels={'memory_mb': 'Memory (MB)', 'step': 'Pipeline Step'},
            markers=True
        )
        fig_memory.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_memory, use_container_width=True)
        
        # Detailed step breakdown
        st.subheader("📊 Detailed Step Analysis")
        
        for i, step in enumerate(pipeline_steps):
            with st.expander(f"Step {i+1}: {step['step']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Processing Time", f"{step['time_ms']:.2f} ms")
                col2.metric("Output Size", step['output_size'])
                col3.metric("Memory Usage", f"{step['memory_mb']:.1f} MB")
                
                # Show step-specific details
                if step['step'] == 'Tokenization':
                    st.json({
                        'word_count': tokens_perf['word_count'],
                        'sentence_count': tokens_perf['sentence_count'],
                        'char_count': tokens_perf['char_count'],
                        'paragraph_count': tokens_perf.get('paragraph_count', 0)
                    })
                elif step['step'] == 'POS Tagging':
                    pos_counts = Counter([tag for _, tag in pos_tags_perf])
                    st.json(dict(pos_counts))
                elif step['step'] == 'Named Entity Recognition':
                    entity_summary = {k: len(v) for k, v in entities_perf.items() if v}
                    st.json(entity_summary if entity_summary else {"No entities found": 0})
                elif step['step'] == 'Sentiment Analysis':
                    st.json({
                        'sentiment': sentiment_perf['sentiment'],
                        'score': round(sentiment_perf['score'], 4),
                        'positive_words': sentiment_perf['positive'],
                        'negative_words': sentiment_perf['negative']
                    })
                elif step['step'] == 'Text Statistics':
                    st.json({
                        'lexical_diversity': round(stats_perf['lexical_diversity'], 4),
                        'avg_words_per_sentence': round(stats_perf['avg_words_per_sentence'], 2),
                        'avg_chars_per_word': round(stats_perf['avg_chars_per_word'], 2),
                        'flesch_reading_ease': round(stats_perf['flesch_reading_ease'], 2),
                        'perplexity': round(stats_perf['perplexity'], 2)
                    })
                elif step['step'] == 'Keyword Extraction':
                    keyword_dict = {word: round(score, 4) for word, score in keywords_perf[:10]}
                    st.json(keyword_dict)
        
        # Model information section
        st.subheader("🤖 Model & System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **NLP Pipeline Configuration:**
            - Tokenizer: Regular Expression Based
            - POS Tagger: Pattern Matching
            - NER: Pattern-based Recognition
            - Sentiment: Lexicon-based Analysis
            - Keywords: TF-IDF Vectorization
            - Statistics: Custom Readability Metrics
            """)
        
        with col2:
            st.markdown(f"""
            **System Resources:**
            - CPU Cores: {psutil.cpu_count()}
            - Total Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB
            - Available Memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB
            - Python Version: {psutil.sys.version.split()[0]}
            - Process ID: {psutil.os.getpid()}
            """)
        
        # Raw data inspection
        st.subheader("🔍 Raw Data Inspection")
        
        inspection_type = st.selectbox(
            "Select data to inspect:",
            ["Tokenization Output", "POS Tags", "Named Entities", "Sentiment Scores", "Text Statistics", "Keywords"]
        )
        
        if inspection_type == "Tokenization Output":
            st.json({
                'word_tokens': tokens_perf.get('word_tokens', [])[:50],  # First 50 tokens
                'sentences': tokens_perf.get('sentences', [])[:10],      # First 10 sentences
                'metadata': {
                    'total_words': tokens_perf.get('word_count', 0),
                    'total_sentences': tokens_perf.get('sentence_count', 0),
                    'total_chars': tokens_perf.get('char_count', 0)
                }
            })
        elif inspection_type == "POS Tags":
            st.json([{'word': word, 'tag': tag} for word, tag in pos_tags_perf[:50]])
        elif inspection_type == "Named Entities":
            st.json(entities_perf)
        elif inspection_type == "Sentiment Scores":
            st.json(sentiment_perf)
        elif inspection_type == "Text Statistics":
            st.json(stats_perf)
        elif inspection_type == "Keywords":
            st.json([{'keyword': word, 'score': score} for word, score in keywords_perf])

def display_privacy_assistant(policy_text, analysis_results):
    """Display the AI-powered privacy assistant interface"""
    st.markdown("---")
    st.subheader("🤖 AI Privacy Assistant")
    
    # Initialize assistant
    assistant = PrivacyAssistant()
    
    if not assistant.is_available():
        st.warning("""
        🔑 **AI Assistant Setup Required**
        
        The AI assistant requires an OpenAI API key to function. To enable this feature:
        
        1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Copy `.env.example` to `.env` 
        3. Add your API key to the `.env` file
        4. Restart the application
        
        The AI assistant helps answer questions about privacy policies in plain language!
        """)
        return
    
    # Create context for the assistant
    context = assistant.analyze_policy_with_context(policy_text, analysis_results)
    
    # Risk assessment
    st.markdown("#### 🚨 Privacy Risk Assessment")
    risk_assessment = assistant.explain_privacy_risks(analysis_results)
    st.markdown(f"""
    <div class="card">
        <pre style="white-space: pre-wrap; font-family: inherit;">{risk_assessment}</pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggested questions
    st.markdown("#### 💭 Suggested Questions")
    suggested_questions = assistant.get_suggested_questions(analysis_results)
    
    # Create columns for suggested questions
    cols = st.columns(2)
    for i, question in enumerate(suggested_questions):
        col = cols[i % 2]
        if col.button(f"❓ {question}", key=f"suggestion_{i}"):
            # Auto-fill the question in the text input
            st.session_state.ai_question = question
    
    # Question input
    st.markdown("#### 💬 Ask the AI Assistant")
    
    # Initialize question state
    if 'ai_question' not in st.session_state:
        st.session_state.ai_question = ""
    
    question = st.text_input(
        "Ask a question about this privacy policy:",
        value=st.session_state.ai_question,
        placeholder="e.g., What personal data does this app collect?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Ask Assistant", type="primary", disabled=not question.strip()):
            if question.strip():
                with st.spinner("🤔 AI is thinking..."):
                    response = assistant.ask_question(question, context)
                
                st.markdown("#### 🤖 Assistant Response")
                st.markdown(f"""
                <div class="card">
                    <h5>❓ Question:</h5>
                    <p><em>{question}</em></p>
                    <h5>💡 Answer:</h5>
                    <p>{response}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Clear the question
                st.session_state.ai_question = ""
    
    with col2:
        if st.button("🧹 Clear Question"):
            st.session_state.ai_question = ""
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear History"):
            assistant.clear_conversation_history()
            st.success("Chat history cleared!")
    
    # Conversation history
    history = assistant.get_conversation_history()
    if history:
        st.markdown("#### 📝 Recent Conversation")
        
        # Show only last 3 exchanges
        recent_history = history[-3:] if len(history) > 3 else history
        
        for i, exchange in enumerate(reversed(recent_history)):
            with st.expander(f"Exchange {len(recent_history) - i}: {exchange['question'][:50]}..."):
                st.markdown(f"""
                **Question:** {exchange['question']}
                
                **Response:** {exchange['response']}
                """)

def main():
    """Main Streamlit application"""
    
    # Initialize session state for text content
    if 'text_content' not in st.session_state:
        st.session_state.text_content = None
    if 'text_source' not in st.session_state:
        st.session_state.text_source = None
    
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
    page = st.sidebar.selectbox(
        "Choose Page:",
        ["🔐 Privacy Policy Analysis", "📊 Text Comparison", "🤓 Stats for Nerds"]
    )
    
    if page == "📊 Text Comparison":
        display_text_comparison_tool()
        return
    elif page == "🤓 Stats for Nerds":
        display_stats_for_nerds()
        return
    
    # Original privacy policy analysis section
    analysis_type = st.sidebar.radio(
        "Choose Input Method:",
        ["📄 Upload PDF", "🔗 Enter URL", "📝 Paste Text"]
    )
    
    # Load model
    model_package = load_model()
    if model_package is None:
        display_model_setup_instructions()
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Main content area
    if analysis_type == "📄 Upload PDF":
        st.subheader("📄 Upload Privacy Policy PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                
            if extracted_text:
                st.session_state.text_content = extracted_text
                st.session_state.text_source = "PDF Upload"
                st.success(f"✅ Text extracted! ({len(extracted_text.split())} words)")
                
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted Text", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)
    
    elif analysis_type == "🔗 Enter URL":
        st.subheader("🔗 Enter Privacy Policy URL")
        url = st.text_input("Privacy Policy URL", placeholder="https://example.com/privacy-policy")
        
        if st.button("Extract Text from URL") and url:
            with st.spinner("Extracting text from URL..."):
                extracted_text = extract_text_from_url(url)
            
            if extracted_text:
                st.session_state.text_content = extracted_text
                st.session_state.text_source = f"URL: {url}"
                st.success(f"✅ Text extracted! ({len(extracted_text.split())} words)")
                
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted Text", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)
    
    elif analysis_type == "📝 Paste Text":
        st.subheader("📝 Paste Privacy Policy Text")
        text_input = st.text_area("Privacy Policy Text", height=200, placeholder="Paste the privacy policy text here...")
        
        if text_input:
            st.session_state.text_content = text_input
            st.session_state.text_source = "Manual Input"
            st.success(f"✅ Text ready for analysis! ({len(text_input.split())} words)")
    
    # Clear content button
    if st.session_state.text_content:
        if st.button("🗑️ Clear Content"):
            st.session_state.text_content = None
            st.session_state.text_source = None
            st.rerun()
    
    # Show current content status
    if st.session_state.text_content:
        st.info(f"📄 Content loaded from: {st.session_state.text_source} ({len(st.session_state.text_content.split())} words)")
    
    # Analysis section
    if st.session_state.text_content and len(st.session_state.text_content.strip()) > 50:
        st.markdown("---")
        
        if st.button("🔍 Analyze Privacy Policy", type="primary"):
            with st.spinner("Analyzing privacy policy..."):
                
                # Classification
                st.subheader("📋 Quick Summary")
                summary = summarize_text(st.session_state.text_content, model_package)
                st.markdown(f"""
                <div class="card">
                    <h4>📝 Policy Summary</h4>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Classification results
                classification_results = classify_privacy_policy(st.session_state.text_content, model_package)
                
                if classification_results:
                    display_classification_results(classification_results)
                    
                    # Vague language analysis
                    st.markdown("---")
                    vague_indicators = analyze_vague_language(st.session_state.text_content)
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
                    
                    # AI Privacy Assistant Section
                    if PrivacyAssistant:
                        display_privacy_assistant(st.session_state.text_content, classification_results)
                else:
                    st.error("❌ Classification failed. Please check the model and try again.")
    
    elif st.session_state.text_content and len(st.session_state.text_content.strip()) <= 50:
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