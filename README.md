# Lyllium - Women's Health Assistant

A RAG (Retrieval-Augmented Generation) application providing specialized women's health information through an AI assistant named Lyllium.

## Overview

Lyllium is designed to help users with women's health topics including:
- Hormones and hormonal health
- Fertility and reproductive wellness
- PCOS, endometriosis, and other conditions
- Menopause and perimenopause
- Pregnancy and postpartum care
- Breast health
- Women-specific aspects of conditions like osteoporosis and heart health

## Features

- **Specialized Knowledge Base**: Curated women's health information
- **Contextual Responses**: RAG pipeline provides relevant, source-backed answers
- **Friendly Interface**: Conversational AI assistant with warm, supportive tone
- **Safety Boundaries**: Appropriate redirects for out-of-scope questions and adds an evaluation report
- **Medical Disclaimers**: Reminds users to consult healthcare providers for personalized advice

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd lyllium-health-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
# Add other configuration variables as needed
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501` for Streamlit)

3. Ask Lyllium questions about women's health topics

## Project Structure

```
project/
├── app.py                 # Main application file
├── static/               # CSS and static assets
│   └── style.css
├── knowledge/                 # Knowledge base files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Knowledge Base

The knowledge base contains curated information about women's health topics. To add new content:

1. Place documents in the `knowledge/` folder
2. Run the ingestion script to update the vector database
3. Test responses to ensure quality


## Response Format

The system uses this input format:
```
**Knowledge base context:** {context}
**User's question:** {prompt}
```
## Disclaimer

This application provides general health information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.