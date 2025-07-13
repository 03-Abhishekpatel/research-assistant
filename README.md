# 📚 Smart Research Assistant

An AI-powered tool that helps researchers and students quickly analyze documents, answer questions, and test comprehension through interactive challenges.

## 🚀 Features

- **Document Processing**: Upload PDF or TXT files for analysis
- **AI Summarization**: Automatic 150-word summary generation
- **Two Interaction Modes**:
  - **Ask Anything**: Get answers to your document-specific questions
  - **Challenge Me**: Test your understanding with generated questions
- **Reference Tracking**: Answers include source references from the document
- **Secure API Handling**: Environment variable protection for your OpenAI key

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3.8+
- **NLP Processing**: OpenAI API
- **Document Handling**: PyPDF2 (for PDFs), Text processing
- **Environment Management**: python-dotenv

## ⚙️ Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (get from [platform.openai.com](https://platform.openai.com))
- Git (for version control)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant