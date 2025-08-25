#!/usr/bin/env python3
"""
Advanced NLP Concepts Module
Implements various NLP techniques with visual demonstrations
"""

import re
import string
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class NLPAnalyzer:
    """Comprehensive NLP Analysis Class"""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'you', 'your', 'we', 'our',
            'this', 'these', 'they', 'them', 'their', 'have', 'had', 'can',
            'may', 'could', 'would', 'should', 'or', 'but', 'if', 'when',
            'where', 'how', 'what', 'who', 'which', 'why', 'do', 'did',
            'does', 'not', 'no', 'nor', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own'
        }
        
    def basic_tokenization(self, text):
        """Demonstrate different types of tokenization"""
        results = {}
        
        # Word tokenization (simple)
        words = re.findall(r'\b\w+\b', text.lower())
        results['word_tokens'] = words
        results['word_count'] = len(words)
        
        # Sentence tokenization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        results['sentences'] = sentences
        results['sentence_count'] = len(sentences)
        
        # Character tokenization
        results['char_count'] = len(text)
        results['char_count_no_spaces'] = len(text.replace(' ', ''))
        
        # Paragraph tokenization
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        results['paragraphs'] = paragraphs
        results['paragraph_count'] = len(paragraphs)
        
        return results
    
    def pos_tagging_simple(self, text):
        """Simple POS tagging using pattern matching"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Simple POS patterns
        pos_patterns = {
            'NOUN': [r'.*tion$', r'.*ment$', r'.*ness$', r'.*ity$', r'.*data$', 
                     r'.*information$', r'.*policy$', r'.*service$', r'.*user$'],
            'VERB': [r'.*ing$', r'.*ed$', r'collect', r'share', r'use', 
                     r'provide', r'obtain', r'store', r'process'],
            'ADJ': [r'.*ful$', r'.*less$', r'.*able$', r'personal', r'private',
                    r'secure', r'necessary', r'appropriate'],
            'MODAL': [r'may', r'might', r'could', r'would', r'should', r'can'],
            'PREP': [r'in', r'on', r'at', r'by', r'for', r'with', r'from', r'to']
        }
        
        tagged_words = []
        for word in words:
            tag = 'NOUN'  # Default
            for pos, patterns in pos_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, word):
                        tag = pos
                        break
                if tag != 'NOUN':
                    break
            tagged_words.append((word, tag))
        
        return tagged_words
    
    def named_entity_recognition(self, text):
        """Simple NER using pattern matching"""
        entities = {
            'EMAIL': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'URL': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            'PHONE': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
            'ORGANIZATION': [],  # Would need more sophisticated matching
            'LEGAL_TERM': re.findall(r'\b(?:privacy policy|terms of service|data protection|gdpr|ccpa|cookies|consent)\b', text.lower())
        }
        
        # Find potential organization names (capitalized sequences)
        org_pattern = r'\b(?:[A-Z][a-z]+ )*[A-Z][a-z]+(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation))?\b'
        entities['ORGANIZATION'] = re.findall(org_pattern, text)
        
        return entities
    
    def sentiment_analysis_simple(self, text):
        """Simple sentiment analysis using word lists"""
        positive_words = {
            'secure', 'protect', 'safe', 'privacy', 'consent', 'choice', 'control',
            'transparent', 'respect', 'rights', 'delete', 'opt-out', 'clear'
        }
        
        negative_words = {
            'share', 'sell', 'disclose', 'transfer', 'vague', 'unclear', 'may',
            'might', 'could', 'discretion', 'necessary', 'appropriate', 'partners'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return {'sentiment': 'neutral', 'score': 0, 'positive': 0, 'negative': 0}
        
        pos_ratio = positive_count / total_words
        neg_ratio = negative_count / total_words
        
        score = pos_ratio - neg_ratio
        
        if score > 0.02:
            sentiment = 'positive'
        elif score < -0.02:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive': positive_count,
            'negative': negative_count,
            'positive_ratio': pos_ratio,
            'negative_ratio': neg_ratio
        }
    
    def text_statistics(self, text):
        """Comprehensive text statistics"""
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic stats
        stats = {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_chars_per_word': sum(len(word) for word in words) / len(words) if words else 0
        }
        
        # Vocabulary richness
        unique_words = set(words)
        stats['unique_words'] = len(unique_words)
        stats['lexical_diversity'] = len(unique_words) / len(words) if words else 0
        
        # Readability (simplified Flesch Reading Ease)
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            stats['flesch_reading_ease'] = max(0, min(100, flesch_score))
        else:
            stats['flesch_reading_ease'] = 0
        
        # Perplexity score
        stats['perplexity'] = self.calculate_perplexity(text)
        
        return stats
    
    def _count_syllables(self, word):
        """Simple syllable counting"""
        vowels = 'aeiouy'
        word = word.lower()
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def calculate_perplexity(self, text):
        """Calculate perplexity score using n-gram language model"""
        import math
        from collections import defaultdict
        
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < 3:
            return 0.0  # Not enough text for meaningful perplexity
        
        # Add sentence boundary markers
        words = ['<s>'] + words + ['</s>']
        
        # Build bigram and unigram counts
        unigram_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        
        for i, word in enumerate(words):
            unigram_counts[word] += 1
            if i > 0:
                bigram = (words[i-1], word)
                bigram_counts[bigram] += 1
        
        # Calculate perplexity using bigram model with add-1 smoothing
        log_prob_sum = 0.0
        n_bigrams = 0
        vocabulary_size = len(unigram_counts)
        
        for i in range(1, len(words)):
            prev_word = words[i-1]
            curr_word = words[i]
            bigram = (prev_word, curr_word)
            
            # Add-1 smoothing
            bigram_count = bigram_counts[bigram]
            unigram_count = unigram_counts[prev_word]
            
            # Smoothed probability: (count + 1) / (context_count + vocabulary_size)
            smoothed_prob = (bigram_count + 1) / (unigram_count + vocabulary_size)
            
            log_prob_sum += math.log2(smoothed_prob)
            n_bigrams += 1
        
        if n_bigrams == 0:
            return 0.0
        
        # Perplexity = 2^(-1/N * sum(log2(P(w_i))))
        avg_log_prob = log_prob_sum / n_bigrams
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity
    
    def keyword_extraction_tfidf(self, text, top_k=10):
        """Extract keywords using TF-IDF"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        
        if len(sentences) < 2:
            # If too few sentences, use word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_freq = Counter(words)
            return [(word, freq) for word, freq in word_freq.most_common(top_k)]
        
        vectorizer = TfidfVectorizer(
            stop_words=list(self.stop_words),
            max_features=100,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Sort by score
            indices = np.argsort(mean_scores)[::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in indices[:top_k]]
            
            return keywords
        except:
            # Fallback to word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_freq = Counter(words)
            return [(word, freq) for word, freq in word_freq.most_common(top_k)]
    
    def ngram_analysis(self, text, n=2, top_k=10):
        """Analyze n-grams in text"""
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        if len(words) < n:
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        ngram_freq = Counter(ngrams)
        return ngram_freq.most_common(top_k)
    
    def topic_modeling_simple(self, text, n_topics=3):
        """Simple topic modeling using LDA"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 30]
        
        if len(sentences) < n_topics:
            return []
        
        try:
            vectorizer = CountVectorizer(
                stop_words=list(self.stop_words),
                max_features=50,
                min_df=1
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            if doc_term_matrix.shape[1] < n_topics:
                n_topics = min(doc_term_matrix.shape[1], 2)
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words[:5],
                    'weights': [topic[i] for i in top_words_idx[:5]]
                })
            
            return topics
        except:
            return []
    
    def text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        try:
            vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback to simple word overlap
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if len(union) == 0:
                return 0.0
            
            return len(intersection) / len(union)
    
    def create_word_frequency_chart(self, text, top_k=15):
        """Create word frequency data for visualization"""
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        word_freq = Counter(words)
        top_words = word_freq.most_common(top_k)
        
        if not top_words:
            return None
        
        return pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    def create_ngram_chart(self, text, n=2, top_k=10):
        """Create n-gram frequency data for visualization"""
        ngrams = self.ngram_analysis(text, n, top_k)
        
        if not ngrams:
            return None
        
        return pd.DataFrame(ngrams, columns=[f'{n}-gram', 'Frequency'])
    
    def analyze_privacy_patterns(self, text):
        """Analyze privacy-specific patterns in text"""
        patterns = {
            'data_collection_verbs': [
                'collect', 'gather', 'obtain', 'acquire', 'receive', 'record',
                'capture', 'harvest', 'compile', 'accumulate'
            ],
            'data_sharing_verbs': [
                'share', 'disclose', 'provide', 'transfer', 'transmit', 'send',
                'distribute', 'release', 'reveal', 'convey'
            ],
            'vague_terms': [
                'may', 'might', 'could', 'possible', 'reasonable', 'appropriate',
                'necessary', 'discretion', 'time to time', 'as needed'
            ],
            'user_rights': [
                'opt-out', 'delete', 'remove', 'choice', 'control', 'access',
                'correct', 'update', 'withdraw', 'consent'
            ]
        }
        
        text_lower = text.lower()
        results = {}
        
        for category, terms in patterns.items():
            found_terms = []
            for term in terms:
                count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if count > 0:
                    found_terms.append({'term': term, 'count': count})
            
            results[category] = {
                'terms': found_terms,
                'total_count': sum(item['count'] for item in found_terms)
            }
        
        return results

    def language_complexity_analysis(self, text):
        """Analyze the complexity of language used"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return {}
        
        # Word length distribution
        word_lengths = [len(word) for word in words]
        
        # Complex word indicators (words with 3+ syllables or 7+ characters)
        complex_words = [word for word in words if len(word) >= 7 or self._count_syllables(word) >= 3]
        
        # Legal/technical jargon
        jargon_terms = [
            'pursuant', 'heretofore', 'aforementioned', 'notwithstanding',
            'hereinafter', 'thereof', 'whereby', 'therein', 'compliance',
            'accordance', 'aggregate', 'utilize', 'facilitate', 'implement'
        ]
        
        found_jargon = [term for term in jargon_terms if term in words]
        
        return {
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'complex_word_ratio': len(complex_words) / len(words),
            'complex_words': complex_words[:10],  # Show first 10
            'jargon_terms': found_jargon,
            'jargon_ratio': len(found_jargon) / len(words) if words else 0
        }