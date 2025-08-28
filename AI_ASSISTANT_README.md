# 🤖 AI Privacy Assistant - Setup Guide

## Overview

The AI Privacy Assistant adds conversational capabilities to the existing privacy policy analyzer, allowing users to ask natural language questions about privacy policies and receive context-aware responses.

## Features

- 🤖 **Conversational Interface**: Ask questions in plain English
- 🎯 **Context-Aware Responses**: Uses existing OPP-115 analysis results as context
- 🚨 **Privacy Risk Assessment**: Automatic risk evaluation with clear explanations
- 💭 **Suggested Questions**: Dynamic question suggestions based on policy analysis
- 📝 **Conversation History**: Maintains session memory for follow-up questions
- 🔐 **Secure Configuration**: API keys managed via environment variables

## Setup Instructions

### 1. Install Dependencies

```bash
pip install openai python-dotenv
```

### 2. Configure OpenAI API

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### 3. Test the Installation

```bash
python test_privacy_assistant.py
```

## Usage

### In the Streamlit App

1. Upload a privacy policy (PDF, URL, or text)
2. Click "🔍 Analyze Privacy Policy" 
3. Scroll down to the "🤖 AI Privacy Assistant" section
4. Use suggested questions or ask your own

### Example Questions

- "What personal data does this policy collect?"
- "Can I delete my personal information?"
- "Does this app share data with third parties?"
- "How long do they keep my data?"
- "What are the biggest privacy risks?"

### Programmatic Usage

```python
from privacy_assistant import PrivacyAssistant

# Initialize assistant
assistant = PrivacyAssistant()

# Check if API is configured
if assistant.is_available():
    # Ask a question with context
    response = assistant.ask_question(
        "What data do they collect?", 
        policy_context
    )
    print(response)
```

## Configuration Options

Environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=500
OPENAI_TEMPERATURE=0.3
```

## Privacy & Security

- 🔐 API keys are stored securely in environment variables
- 🛡️ No sensitive user data is sent to OpenAI beyond the policy text
- ⚖️ Data minimization principles applied
- 🚫 Users can opt-out by not configuring API keys

## Error Handling

The assistant gracefully handles:
- Missing or invalid API keys
- Rate limiting
- Network errors
- API unavailability

When the AI assistant is unavailable, all other features continue to work normally.

## Cost Considerations

- Uses GPT-3.5-turbo by default (cost-effective)
- Responses limited to 500 tokens
- Conversation history limited to recent exchanges
- Monitor usage in your OpenAI dashboard