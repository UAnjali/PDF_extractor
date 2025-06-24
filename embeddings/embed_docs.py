import os
import sys
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import tempfile
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pathlib import Path
from config.config import load_config

# Set up logging
logging.basicConfig(
    filename='embedding.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_data_dir() -> Path:
    """Get the data directory path."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def copy_uploaded_file(uploaded_file, target_dir: Path) -> Optional[Path]:
    """Copy an uploaded file to the data directory."""
    try:
        # Create a unique filename to avoid conflicts
        target_path = target_dir / uploaded_file.name
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"Successfully copied uploaded file to {target_path}")
        return target_path
    except Exception as e:
        logging.error(f"Error copying uploaded file: {str(e)}")
        return None

def delete_document(filename: str) -> bool:
    """Delete a document from the data directory and re-embed remaining documents."""
    try:
        data_dir = get_data_dir()
        file_path = data_dir / filename
        
        if not file_path.exists():
            logging.warning(f"File {filename} not found in data directory")
            return False
            
        # Delete the file
        os.remove(file_path)
        logging.info(f"Successfully deleted {filename}")
        
        # Re-embed all remaining documents
        success = reembed_all_documents()
        if success:
            logging.info("Successfully re-embedded remaining documents")
        else:
            logging.error("Failed to re-embed remaining documents")
            
        return success
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        return False

def get_available_documents() -> List[str]:
    """Get list of available documents in the data directory."""
    data_dir = get_data_dir()
    return [f.name for f in data_dir.glob("*") if f.is_file()]

def load_documents_from_file(file_path: Path) -> List:
    """Load documents from a file based on its type."""
    try:
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
            logging.info(f"Loading PDF file: {file_path}")
        elif file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path))
            logging.info(f"Loading text file: {file_path}")
        else:
            logging.warning(f"Unsupported file type: {file_path.suffix}")
            return []

        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logging.error(f"Error loading documents from {file_path}: {str(e)}")
        raise

def load_documents() -> List:
    """Load all documents from the data directory."""
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            logging.info("Created data directory")
            return []

        all_documents = []
        for file_path in data_dir.glob("*"):
            if file_path.suffix.lower() in ['.pdf', '.txt']:
                documents = load_documents_from_file(file_path)
                all_documents.extend(documents)
                logging.info(f"Added {len(documents)} documents from {file_path}")

        logging.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    except Exception as e:
        logging.error(f"Error loading documents: {str(e)}")
        raise

def process_uploaded_file(file_path: Path) -> bool:
    try:
        # Load the document
        loader = TextLoader(str(file_path))
        documents = loader.load()
        
        # Add metadata to each document
        for doc in documents:
            # Extract course type from filename or content
            course_type = "Data Analytics"  # You can make this dynamic based on file content or name
            doc.metadata.update({
                "course_type": course_type,
                "source": file_path.name,
                "processed_date": datetime.now().isoformat()
            })
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)
        
        # Log the number of chunks created
        logging.info(f"Created {len(splits)} chunks from {file_path.name}")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=load_config()["OPENAI_API_KEY"],
            dimensions=3072
        )
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=load_config()["QDRANT_URL"],
            api_key=load_config()["QDRANT_API_KEY"]
        )
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="sales_counsellor",
            embedding=embeddings
        )
        
        # Add documents to vector store with metadata
        vector_store.add_documents(splits)
        logging.info(f"Successfully added {len(splits)} chunks from {file_path.name} to vector store with metadata")
        
        return True
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return False

def reembed_all_documents() -> bool:
    """Re-embed all documents in the data directory."""
    try:
        # Load configuration
        config = load_config()

        # Initialize embeddings with OpenAI text-embedding-3-large
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=config["OPENAI_API_KEY"],
            dimensions=3072  # Using the larger dimension size for better performance
        )

        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=config["QDRANT_URL"],
            api_key=config["QDRANT_API_KEY"]
        )

        # Delete existing collection
        try:
            qdrant_client.delete_collection("sales_counsellor")
            logging.info("Deleted existing collection to ensure consistency")
        except Exception as e:
            logging.info(f"No existing collection to delete: {str(e)}")

        # Create new collection with correct dimensions for text-embedding-3-large
        qdrant_client.create_collection(
            collection_name="sales_counsellor",
            vectors_config={
                "size": 3072,  # Size for text-embedding-3-large
                "distance": "Cosine"
            }
        )
        logging.info("Created new collection 'sales_counsellor' with correct dimensions")

        # Load all documents from data directory
        all_documents = []
        data_dir = Path("data")
        for doc_path in data_dir.glob("*"):
            if doc_path.suffix.lower() in ['.pdf', '.txt']:
                docs = load_documents_from_file(doc_path)
                all_documents.extend(docs)
                logging.info(f"Loaded {len(docs)} documents from {doc_path}")

        if not all_documents:
            logging.error("No documents found to embed")
            return False

        # Split all documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(all_documents)
        logging.info(f"Split {len(all_documents)} documents into {len(splits)} chunks")

        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="sales_counsellor",
            embedding=embeddings
        )

        # Add all documents to vector store
        vector_store.add_documents(splits)
        logging.info(f"Successfully re-embedded all {len(splits)} chunks")

        return True
    except Exception as e:
        logging.error(f"Error re-embedding all documents: {str(e)}")
        return False

if __name__ == "__main__":
    reembed_all_documents()
