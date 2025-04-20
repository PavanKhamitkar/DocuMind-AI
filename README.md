# DocuMind AI

An advanced document analysis system powered by Google's Gemini AI and FAISS vector database. Extract insights, analyze content, and interact with your PDF documents using state-of-the-art natural language processing.

## 🚀 Features

- **PDF Document Analysis**: Upload and process PDF documents seamlessly
- **Advanced RAG Architecture**: Utilizing Retrieval-Augmented Generation for precise answers
- **Vector Search**: FAISS-powered semantic search capabilities
- **Interactive UI**: Clean and responsive Streamlit interface
- **Conversation History**: Track and review previous Q&A interactions
- **Document Preview**: Quick preview of uploaded documents
- **Enterprise-Grade Security**: Built-in security headers and API key validation

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI/ML**:
  - Google Gemini AI
  - LangChain Framework
  - FAISS Vector Database
- **Core**: Python 3.9+
- **Key Libraries**:
  - `langchain`: For RAG implementation
  - `langchain-google-genai`: Gemini AI integration
  - `faiss-cpu`: Vector similarity search
  - `PyPDF2`: PDF processing
  - `python-dotenv`: Environment management

## 📊 System Architecture

```plaintext
User Input → PDF Processing → Text Chunking → Vector Embedding → FAISS DB
                  ↓                                    ↓
            Document Preview                    Semantic Search
                  ↓                                    ↓
            RAG Processing ← Gemini AI ← Query Processing
                  ↓
            Response Generation
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/PavanKhamitkar/DocuMind-AI.git
cd documind-ai
```

2. **Set up environment**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure API Key**
```bash
# Create .env file
echo GEMINI_API_KEY=your_key_here > .env
```

4. **Run the application**
```bash
streamlit run main.py
```

## 💻 Usage

1. Launch the application
2. Upload a PDF document
3. Wait for processing completion
4. Enter your questions about the document
5. Receive AI-powered analytical responses

## 🔧 Configuration

Customize the system by modifying `CONFIG` in `main.py`:

```python
CONFIG = {
    "embedding_model": "models/embedding-001",
    "llm_model": "gemini-1.5-flash",
    "chunk_size": 10000,
    "chunk_overlap": 1000,
    "max_output_tokens": 2048,
    "temperature": 0.3,
    "search_kwargs": {"k": 3}
}
```

## 🛡️ Security Features

- API Key validation
- Content Security Policy headers
- Secure file handling
- Environment variable protection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Contact

Pavan Khamitkar - [khamitkarpavan8@gmail.com](mailto:khamitkarpavan8@gmail.com)

Project Link: [https://github.com/PavanKhamitkar/DocuMind-AI](https://github.com/PavanKhamitkar/DocuMind-AI)