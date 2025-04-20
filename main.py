# -*- coding: utf-8 -*-
"""
Advanced Document Analysis System with Gemini AI and FAISS

A Retrieval-Augmented Generation (RAG) system for PDF document analysis.

Copyright (c) 2024 DocuMind AI
Licensed under the MIT License
"""

import os
import warnings
from datetime import datetime
import tempfile
from dotenv import load_dotenv
import streamlit as st

# LangChain components
from langchain import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Configuration
warnings.filterwarnings("ignore")
load_dotenv()

# Constants
CONFIG = {
    "embedding_model": "models/embedding-001",
    "llm_model": "gemini-1.5-flash",
    "chunk_size": 10000,
    "chunk_overlap": 1000,
    "max_output_tokens": 2048,
    "temperature": 0.3,
    "search_kwargs": {"k": 3}
}

class DocumentAnalyzer:
    """Core class for document analysis functionality."""
    
    def __init__(self):
        self.qa_chain = None
        self.vector_db = None
        self._validate_api_key()
        
    def _validate_api_key(self):
        """Validate the Gemini API key."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables.")

    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded PDF file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file.name

    def load_and_split_document(self, file_path):
        """Load and split the document into chunks."""
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        full_text = "\n\n".join([p.page_content for p in pages])
        return text_splitter.split_text(full_text), pages[0].page_content

    def initialize_vector_db(self, text_chunks):
        """Initialize the FAISS vector database."""
        embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG["embedding_model"],
            google_api_key=self.api_key
        )
        self.vector_db = FAISS.from_texts(text_chunks, embeddings)

    def initialize_qa_chain(self):
        """Initialize the QA chain."""
        llm = ChatGoogleGenerativeAI(
            model=CONFIG["llm_model"],
            google_api_key=self.api_key,
            temperature=CONFIG["temperature"],
            max_output_tokens=CONFIG["max_output_tokens"]
        )
        
        prompt_template = """As a professional document analyst, provide a thorough answer based on the context.
        If the answer isn't clear from the context, state that explicitly.
        Conclude with "Please let me know if you need further clarification."

        Context: {context}
        Question: {question}

        Analytical Answer:"""
        
        qa_prompt = PromptTemplate.from_template(prompt_template)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_db.as_retriever(search_kwargs=CONFIG["search_kwargs"]),
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt}
        )

    def query_document(self, question):
        """Query the document and return the response."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        return self.qa_chain({"query": question})


class AppInterface:
    """Handles the Streamlit user interface."""
    
    def __init__(self):
        self.analyzer = DocumentAnalyzer()
        self._setup_page_config()
        
    def _setup_page_config(self):
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="DocuMind AI",
            page_icon="📄",
            layout="centered",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <meta http-equiv="Content-Security-Policy" 
            content="default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; 
            style-src 'self' 'unsafe-inline';">
        """, unsafe_allow_html=True)
        
        # Inject custom CSS
        st.markdown("""
        <style>
            .reportview-container .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .stTextInput input {
                border-radius: 8px;
                padding: 12px;
            }
            .stButton button {
                border-radius: 8px;
                background-color: #4f8bf9;
                color: white;
                font-weight: 500;
            }
            .st-expander {
                border-left: 4px solid #4f8bf9;
            }
        </style>
        """, unsafe_allow_html=True)

    def _display_sidebar(self):
        """Display the application sidebar."""
        with st.sidebar:
            st.title("DocuMind AI")
            st.markdown("""
            **Advanced Document Analysis System**
            
            Upload PDF documents and get AI-powered insights.
            Uses Gemini AI with RAG architecture for precise answers.
            """)
            st.markdown("---")
            st.markdown(f"**Version:** 1.0.0")

    def _display_document_preview(self, content):
        """Display a preview of the document content."""
        with st.expander("Document Preview", expanded=False):
            st.text(content[:5000] + "...")

    def _display_conversation_history(self, history):
        """Display the conversation history."""
        with st.expander("Conversation History", expanded=False):
            for i, (question, answer) in enumerate(history):
                st.markdown(f"**Q{i+1}:** {question}")
                st.markdown(f"**A{i+1}:** {answer}")
                st.write("---")

    def run(self):
        """Run the main application."""
        self._display_sidebar()
        
        st.title("📄 DocuMind AI")
        st.caption("Advanced document analysis with AI-powered insights")
        
        # Initialize session state
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Maximum file size: 20MB"
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    # Process document
                    temp_file_path = self.analyzer.process_uploaded_file(uploaded_file)
                    text_chunks, first_page_content = self.analyzer.load_and_split_document(temp_file_path)
                    
                    # Display preview
                    self._display_document_preview(first_page_content)
                    
                    # Initialize systems
                    self.analyzer.initialize_vector_db(text_chunks)
                    self.analyzer.initialize_qa_chain()
                    
                    st.success("Document ready for analysis!")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    return
            
            # Question input
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="What are the key findings in this document?"
            )
            
            if question:
                with st.spinner("Analyzing document..."):
                    try:
                        # Get answer
                        result = self.analyzer.query_document(question)
                        
                        # Update history
                        st.session_state.conversation_history.append(
                            (question, result['result'])
                        )
                        
                        # Display results
                        st.markdown("### Analysis Results")
                        st.write(result['result'])
                        
                        # Show history
                        self._display_conversation_history(st.session_state.conversation_history)
                        
                    except Exception as e:
                        st.error(f"Error analyzing document: {str(e)}")


if __name__ == "__main__":
    app = AppInterface()
    app.run()