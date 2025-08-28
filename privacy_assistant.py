#!/usr/bin/env python3
"""
AI-Powered Privacy Assistant
Provides conversational interface for privacy policy analysis using OpenAI GPT
"""

import os
import openai
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional, Tuple
import time

# Make streamlit import optional for testing
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock session_state for testing
    class MockSessionState:
        def __init__(self):
            self.conversation_history = []
        
        def __contains__(self, key):
            return hasattr(self, key)
        
        def __getitem__(self, key):
            return getattr(self, key)
        
        def __setitem__(self, key, value):
            setattr(self, key, value)
    
    class MockSt:
        session_state = MockSessionState()
    
    st = MockSt()

class PrivacyAssistant:
    """AI-powered conversational privacy assistant"""
    
    def __init__(self):
        """Initialize the Privacy Assistant with OpenAI configuration"""
        # Load environment variables
        load_dotenv()
        
        # Get API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key or self.api_key == 'your_api_key_here':
            self.api_key = None
            self.available = False
        else:
            self.available = True
            self.client = openai.OpenAI(api_key=self.api_key)
        
        # Configuration
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '500'))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.3'))
        
        # Initialize conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # System prompt for privacy-focused responses
        self.system_prompt = """You are a helpful privacy assistant that explains privacy policies in simple, user-friendly language. 

Key guidelines:
- Translate complex legal language into plain English
- Focus on what matters most to users about their data privacy
- Be accurate but avoid giving legal advice
- If you're unsure about something, say so
- Keep responses concise and actionable
- Highlight potential privacy risks clearly
- Use the provided analysis context when available

When users ask about privacy policies, help them understand:
- What data is collected and why
- Who data is shared with
- How long data is kept
- What rights users have
- How to exercise those rights
- Potential privacy risks"""

    def is_available(self) -> bool:
        """Check if the assistant is available (API key configured)"""
        return self.available

    def analyze_policy_with_context(self, policy_text: str, analysis_results: Dict) -> str:
        """Combine existing analysis with policy text for GPT context"""
        if not analysis_results:
            return f"Privacy Policy Text:\n{policy_text[:2000]}..."
        
        # Create context from analysis results
        context_parts = ["Privacy Policy Analysis Context:"]
        
        # Add classification results
        if 'data_collection' in analysis_results:
            collection_status = "detected" if analysis_results['data_collection']['predicted'] else "not clearly indicated"
            collection_prob = analysis_results['data_collection']['probability']
            context_parts.append(f"- Data Collection: {collection_status} (confidence: {collection_prob:.1%})")
        
        if 'data_sharing' in analysis_results:
            sharing_status = "detected" if analysis_results['data_sharing']['predicted'] else "not clearly indicated"
            sharing_prob = analysis_results['data_sharing']['probability']
            context_parts.append(f"- Data Sharing: {sharing_status} (confidence: {sharing_prob:.1%})")
        
        if 'data_storage' in analysis_results:
            storage_status = "addressed" if analysis_results['data_storage']['predicted'] else "not clearly addressed"
            storage_prob = analysis_results['data_storage']['probability']
            context_parts.append(f"- Data Storage: {storage_status} (confidence: {storage_prob:.1%})")
        
        if 'vague_language' in analysis_results:
            vague_status = "contains vague language" if analysis_results['vague_language']['predicted'] else "language is relatively clear"
            vague_prob = analysis_results['vague_language']['probability']
            context_parts.append(f"- Language Clarity: {vague_status} (confidence: {vague_prob:.1%})")
        
        context_parts.append(f"\nPrivacy Policy Text (excerpt):\n{policy_text[:1500]}...")
        
        return "\n".join(context_parts)

    def ask_question(self, question: str, policy_context: str) -> str:
        """Send question to GPT with policy context and return response"""
        if not self.available:
            return ("🚫 AI Assistant unavailable. Please configure your OpenAI API key in the .env file. "
                   "Copy .env.example to .env and add your API key from https://platform.openai.com/api-keys")
        
        try:
            # Prepare messages for the conversation
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{policy_context}\n\nQuestion: {question}"}
            ]
            
            # Add conversation history (last 3 exchanges to maintain context)
            recent_history = st.session_state.conversation_history[-6:] if len(st.session_state.conversation_history) > 6 else st.session_state.conversation_history
            for exchange in recent_history:
                messages.insert(-1, {"role": "user", "content": exchange['question']})
                messages.insert(-1, {"role": "assistant", "content": exchange['response']})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Store in conversation history
            st.session_state.conversation_history.append({
                'question': question,
                'response': assistant_response,
                'timestamp': time.time()
            })
            
            return assistant_response
            
        except openai.AuthenticationError:
            return "🔑 Authentication failed. Please check your OpenAI API key in the .env file."
        except openai.RateLimitError:
            return "⏱️ Rate limit exceeded. Please wait a moment and try again."
        except openai.BadRequestError as e:
            return f"❌ Invalid request: {str(e)}"
        except Exception as e:
            return f"🚫 Error communicating with AI assistant: {str(e)}"

    def get_conversation_history(self) -> List[Dict]:
        """Return chat history for session management"""
        return st.session_state.conversation_history

    def clear_conversation_history(self):
        """Clear the conversation history"""
        st.session_state.conversation_history = []

    def get_suggested_questions(self, analysis_results: Dict) -> List[str]:
        """Generate suggested questions based on analysis results"""
        suggestions = [
            "What personal data does this policy collect?",
            "Can I delete my personal information?",
            "Does this app share data with third parties?",
        ]
        
        if not analysis_results:
            return suggestions
        
        # Add context-specific suggestions
        if analysis_results.get('data_collection', {}).get('predicted', False):
            suggestions.append("What specific types of data are collected and why?")
        
        if analysis_results.get('data_sharing', {}).get('predicted', False):
            suggestions.append("Who exactly do they share my data with?")
            
        if analysis_results.get('vague_language', {}).get('predicted', False):
            suggestions.append("Can you explain the vague terms in simpler language?")
            
        if analysis_results.get('data_storage', {}).get('predicted', False):
            suggestions.append("How long do they keep my data?")
        else:
            suggestions.append("What does this policy say about data retention?")
        
        return suggestions[:6]  # Limit to 6 suggestions

    def explain_privacy_risks(self, analysis_results: Dict) -> str:
        """Generate privacy risk explanation"""
        if not analysis_results:
            return "No analysis results available to assess privacy risks."
        
        risks = []
        
        if analysis_results.get('data_collection', {}).get('predicted', False):
            prob = analysis_results['data_collection']['probability']
            if prob > 0.7:
                risks.append("🔴 High likelihood of extensive data collection")
            else:
                risks.append("🟡 Some data collection detected")
        
        if analysis_results.get('data_sharing', {}).get('predicted', False):
            prob = analysis_results['data_sharing']['probability']
            if prob > 0.7:
                risks.append("🔴 High likelihood of data sharing with third parties")
            else:
                risks.append("🟡 Some data sharing may occur")
        
        if analysis_results.get('vague_language', {}).get('predicted', False):
            prob = analysis_results['vague_language']['probability']
            if prob > 0.7:
                risks.append("🔴 Policy contains significant vague language")
            else:
                risks.append("🟡 Some unclear language detected")
        
        if not analysis_results.get('data_storage', {}).get('predicted', False):
            risks.append("🟡 Data storage and retention policies unclear")
        
        if not risks:
            return "🟢 This policy appears to have relatively clear language with standard privacy practices."
        
        return "Privacy Risk Assessment:\n" + "\n".join(f"• {risk}" for risk in risks)