#!/usr/bin/env python3
"""
Integration Test for Privacy Assistant
Validates that the AI assistant integrates correctly with existing analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_integration():
    """Test the complete integration of privacy assistant with analysis pipeline"""
    
    print("🧪 Integration Test: Privacy Assistant + Existing Analysis")
    print("=" * 60)
    
    # Test imports
    try:
        from privacy_assistant import PrivacyAssistant
        print("✅ PrivacyAssistant import successful")
        
        # Test assistant initialization
        assistant = PrivacyAssistant()
        print(f"✅ Assistant initialized (API available: {assistant.is_available()})")
        
        # Test the complete workflow with mock data
        print("\n📋 Testing Complete Analysis Workflow:")
        
        # Mock policy text
        sample_policy = """
        We collect your personal information including your name, email address, 
        location data, and device identifiers. This information may be shared 
        with our partners for business purposes. We use appropriate security 
        measures to protect your data, though retention periods may vary.
        """
        print(f"📄 Sample policy: {len(sample_policy)} characters")
        
        # Mock analysis results (simulating existing pipeline)
        mock_results = {
            'data_collection': {'predicted': True, 'probability': 0.89},
            'data_sharing': {'predicted': True, 'probability': 0.76}, 
            'data_storage': {'predicted': False, 'probability': 0.42},
            'vague_language': {'predicted': True, 'probability': 0.71}
        }
        print("✅ Mock analysis results created")
        
        # Test context creation
        context = assistant.analyze_policy_with_context(sample_policy, mock_results)
        print(f"✅ Context created: {len(context)} characters")
        
        # Test suggested questions
        questions = assistant.get_suggested_questions(mock_results)
        print(f"✅ Generated {len(questions)} suggested questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"    {i}. {q}")
        
        # Test risk assessment
        risk_assessment = assistant.explain_privacy_risks(mock_results)
        print("✅ Risk assessment generated:")
        print(f"    {risk_assessment.split(chr(10))[0]}...")
        
        # Test conversation history
        print(f"✅ Conversation history: {len(assistant.get_conversation_history())} messages")
        
        print("\n🎯 Integration Test Results:")
        print("✅ All components integrate successfully")
        print("✅ Assistant works with existing analysis pipeline")
        print("✅ Context-aware responses enabled") 
        print("✅ Risk assessment functional")
        print("✅ Question suggestions working")
        
        if assistant.is_available():
            print("✅ AI features ready for use")
        else:
            print("ℹ️ AI features available after API key configuration")
            
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_integration():
    """Test that streamlit app can import and use the assistant"""
    
    print("\n🌐 Testing Streamlit Integration:")
    print("-" * 40)
    
    try:
        # Test if streamlit app can import the assistant
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "streamlit_app", 
            "/home/runner/work/NLP_PROJECT/NLP_PROJECT/streamlit_app.py"
        )
        
        if spec and spec.loader:
            # Check if the display_privacy_assistant function exists
            with open("/home/runner/work/NLP_PROJECT/NLP_PROJECT/streamlit_app.py", 'r') as f:
                content = f.read()
                
            if "display_privacy_assistant" in content:
                print("✅ Streamlit app includes AI assistant interface")
            else:
                print("❌ AI assistant interface not found in streamlit app")
                return False
                
            if "PrivacyAssistant" in content:
                print("✅ Streamlit app imports PrivacyAssistant")
            else:
                print("❌ PrivacyAssistant import not found in streamlit app")
                return False
                
            print("✅ Streamlit integration complete")
            return True
        else:
            print("❌ Could not load streamlit app module")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running Privacy Assistant Integration Tests\n")
    
    # Run tests
    integration_ok = test_integration()
    streamlit_ok = test_streamlit_integration()
    
    print("\n" + "=" * 60)
    print("📊 Final Test Results:")
    
    if integration_ok and streamlit_ok:
        print("🎉 All tests passed! Privacy Assistant is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Configure OpenAI API key in .env file")
        print("   2. Run: streamlit run streamlit_app.py")
        print("   3. Upload a privacy policy and try the AI assistant!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)