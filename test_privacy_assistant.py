#!/usr/bin/env python3
"""
Demo script for Privacy Assistant functionality
Tests the AI assistant with mock data without requiring full streamlit setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from privacy_assistant import PrivacyAssistant

def test_privacy_assistant():
    """Test the Privacy Assistant functionality"""
    
    print("🤖 Privacy Assistant Demo")
    print("=" * 50)
    
    # Initialize assistant
    assistant = PrivacyAssistant()
    
    print(f"✅ Assistant initialized")
    print(f"📡 API Available: {assistant.is_available()}")
    
    if not assistant.is_available():
        print("\n🔑 To enable AI features:")
        print("1. Get API key from https://platform.openai.com/api-keys")
        print("2. Copy .env.example to .env")
        print("3. Add your API key to .env file")
        print("4. Restart the application")
    
    print("\n" + "=" * 50)
    
    # Mock analysis results
    mock_analysis = {
        'data_collection': {'predicted': True, 'probability': 0.85},
        'data_sharing': {'predicted': True, 'probability': 0.72},
        'data_storage': {'predicted': False, 'probability': 0.35},
        'vague_language': {'predicted': True, 'probability': 0.68}
    }
    
    # Test suggested questions
    print("\n💭 Suggested Questions:")
    questions = assistant.get_suggested_questions(mock_analysis)
    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question}")
    
    # Test risk assessment
    print("\n🚨 Privacy Risk Assessment:")
    risk_assessment = assistant.explain_privacy_risks(mock_analysis)
    print(risk_assessment)
    
    # Test context creation
    print("\n📋 Context Analysis:")
    mock_policy = """
    We collect your personal information including name, email, location data, 
    and browsing behavior to provide our services. We may share this information 
    with our partners for business purposes and marketing activities. Data is 
    stored securely but retention periods may vary depending on circumstances.
    """
    
    context = assistant.analyze_policy_with_context(mock_policy, mock_analysis)
    print("Context created for AI assistant (truncated):")
    print(context[:300] + "..." if len(context) > 300 else context)
    
    # Test conversation functionality (mock without API call)
    print("\n💬 Conversation Test:")
    if assistant.is_available():
        print("AI Assistant is ready for questions!")
        print("Example: assistant.ask_question('What data do they collect?', context)")
    else:
        print("Mock response: 'Based on the analysis, this policy collects personal data including names, emails, and location information for service provision and marketing purposes.'")
    
    print("\n✅ Demo completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_privacy_assistant()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)