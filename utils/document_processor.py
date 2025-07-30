import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Replacement for OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq  # Replacement for ChatOpenAI

def process_uploaded_file(file_path: str, file_type: str, groq_client=None):
    """Process uploaded PDF or TXT file and create vector store."""
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store using HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU available
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore, splits
    
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

#def generate_summary(documents: list[Document], groq_client=None) -> str:
def generate_summary(documents: list[Document], groq_client=None) -> str:
    """Generate summary with robust error handling"""
    from langchain.chains.summarize import load_summarize_chain
    import requests

    try:
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            groq_api_key=groq_client.api_key if groq_client else os.getenv("GROQ_API_KEY"),
            timeout=30  # Added timeout
        )
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(documents)[:1500]
    
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to Groq API. Check your internet connection.")
    except Exception as e:
        raise Exception(f"Summary generation failed: {str(e)}")
    """Generate a concise summary of the document using Groq."""
    from langchain.chains.summarize import load_summarize_chain
    
    # Initialize Groq LLM
    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",  # or "llama3-70b-8192"
        groq_api_key=groq_client.api_key if groq_client else os.getenv("GROQ_API_KEY")
    )
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    try:
        summary = chain.run(documents)
        return summary[:1500]  # Increased limit as Groq models are more verbose
    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")