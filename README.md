# Privacy Policy Analyzer 🔐

An AI-powered NLP system for analyzing privacy policies and terms of service documents. This tool helps users understand what data is collected, how it's stored, whether it's shared with third parties, and identifies potentially vague or ambiguous language.

## ✨ New Features

### 🤖 AI-Powered Conversational Assistant

The latest version includes an intelligent chatbot that can answer questions about privacy policies in plain English!

- **🎯 Context-Aware Responses**: Uses existing privacy analysis results to provide accurate answers
- **💬 Natural Language Queries**: Ask questions like "What data do they collect?" or "Can I delete my information?"
- **🚨 Risk Assessment**: Automatic privacy risk evaluation with clear explanations
- **💭 Smart Suggestions**: Dynamic question suggestions based on policy analysis
- **📝 Conversation Memory**: Maintains context across multiple questions

#### Quick Setup for AI Assistant

1. Get an OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
2. Copy `.env.example` to `.env` and add your API key
3. Restart the application

See [`AI_ASSISTANT_README.md`](AI_ASSISTANT_README.md) for detailed setup instructions.

## Core Features

- **📊 Data Collection Analysis**: Identifies what personal information is being collected
- **🤝 Data Sharing Detection**: Detects if data is shared with third parties
- **🏗️ Data Storage Assessment**: Analyzes how data is stored and secured
- **⚠️ Vague Language Detection**: Highlights potentially ambiguous or unclear terms
- **📝 Automatic Summarization**: Generates concise summaries of privacy policies
- **🔗 Multiple Input Methods**: Supports PDF upload, URL extraction, and direct text input

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

**Note**: The training script will automatically download the OPP-115 dataset if available, or use synthetic data as a fallback.

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Upload a PDF**: Upload a privacy policy PDF document
2. **Enter a URL**: Provide a direct link to an online privacy policy
3. **Paste Text**: Copy and paste privacy policy text directly

The system will analyze the document and provide:
- Classification results for each category
- Detailed analysis of vague language
- Key recommendations for users
- A summary of the policy

## Model Training

The system uses a DistilBERT-based multi-label classifier trained on privacy policy text. The training process includes:

- Text preprocessing and tokenization
- Multi-label classification for 4 categories:
  - Data Collection
  - Data Sharing  
  - Data Storage
  - Vague Language
- Automatic fallback to synthetic data if OPP-115 dataset is unavailable

## Technical Details

### Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Framework**: PyTorch + Transformers
- **Frontend**: Streamlit
- **Text Processing**: BeautifulSoup, PyPDF2

### Dependencies
- torch>=1.9.0
- transformers>=4.20.0
- streamlit>=1.20.0
- scikit-learn>=1.0.0
- beautifulsoup4>=4.9.0
- PyPDF2>=3.0.0

## File Structure

```
NLP_PROJECT/
├── train_model.py          # Model training script
├── streamlit_app.py        # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── privacy_model.pkl      # Trained model (generated)
```

## Troubleshooting

### Model Not Found Error
If you see "Model not found" when running the Streamlit app:
1. Make sure you've run `python train_model.py` first
2. Check that `privacy_model.pkl` exists in the project directory
3. Verify all dependencies are installed

### CUDA/GPU Issues
The system automatically detects and uses GPU if available, but falls back to CPU:
- For GPU training: Ensure PyTorch with CUDA support is installed
- For CPU-only: The system will work but training will be slower

### Dataset Download Issues
If the OPP-115 dataset download fails:
- The system will automatically use synthetic training data
- You can manually download the dataset to the `data/` directory if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is intended for educational and research purposes. Please ensure compliance with relevant privacy laws and regulations when analyzing real privacy policies.

## Disclaimer

⚠️ This tool provides automated analysis and should not replace legal advice. Always consult with legal professionals for important privacy policy reviews.