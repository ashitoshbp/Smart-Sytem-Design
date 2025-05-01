import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss

class TextProcessor:
    def __init__(self, data_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextProcessor with the path to the processed data.
        
        Args:
            data_path (str): Path to the processed data (JSON format)
            chunk_size (int): Size of text chunks for embedding
            chunk_overlap (int): Overlap between chunks
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data = None
        self.chunks = []
        self.embeddings = None
        self.embedding_model = None
        self.vector_store = None
        
    def load_data(self):
        """
        Load the processed data from JSON.
        
        Returns:
            List[Dict]: List of incident records
        """
        print(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} records")
        return self.data
    
    def create_chunks(self):
        """
        Create text chunks from the incident data.
        
        Returns:
            List[Dict]: List of chunks with metadata
        """
        if self.data is None:
            self.load_data()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        self.chunks = []
        
        for i, record in enumerate(self.data):
            # Create a text representation of the record
            text = f"Incident ID: {i}\n"
            text += f"Type: {record.get('incident_type', 'Unknown')}\n"
            text += f"Description: {record.get('description', 'No description')}\n"
            text += f"Location: {record.get('location', 'Unknown location')}\n"
            text += f"Status: {record.get('status', 'Unknown status')}\n"
            
            # Add dates if available
            if 'open_date' in record:
                text += f"Open Date: {record['open_date']}\n"
            if 'close_date' in record:
                text += f"Close Date: {record['close_date']}\n"
            if 'duration_hours' in record:
                text += f"Duration (hours): {record.get('duration_hours', 'Unknown')}\n"
                
            # Split the text into chunks
            doc_chunks = text_splitter.create_documents([text], [{"source": i, "record": record}])
            
            # Add chunks to the list
            for chunk in doc_chunks:
                self.chunks.append({
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                })
        
        print(f"Created {len(self.chunks)} chunks from {len(self.data)} records")
        return self.chunks
    
    def initialize_embedding_model(self):
        """
        Initialize the embedding model.
        
        Returns:
            HuggingFaceEmbeddings: Initialized embedding model
        """
        print("Initializing embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model initialized")
        return self.embedding_model
    
    def generate_embeddings(self):
        """
        Generate embeddings for all chunks.
        
        Returns:
            np.ndarray: Array of embeddings
        """
        if not self.chunks:
            self.create_chunks()
            
        if self.embedding_model is None:
            self.initialize_embedding_model()
            
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.embedding_model.embed_documents(texts)
        print(f"Generated {len(self.embeddings)} embeddings")
        return self.embeddings
    
    def create_faiss_index(self):
        """
        Create a FAISS index from the embeddings.
        
        Returns:
            faiss.Index: FAISS index
        """
        if self.embeddings is None:
            self.generate_embeddings()
            
        print("Creating FAISS index...")
        embedding_dim = len(self.embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(self.embeddings).astype('float32'))
        self.vector_store = index
        print(f"Created FAISS index with {index.ntotal} vectors of dimension {embedding_dim}")
        return index
    
    def save_processed_data(self, output_dir: str):
        """
        Save the processed data (chunks and index) to disk.
        
        Args:
            output_dir (str): Directory to save the processed data
        """
        if self.chunks is None or self.vector_store is None:
            raise ValueError("Chunks and vector store must be created before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks
        chunks_path = os.path.join(output_dir, "chunks.json")
        with open(chunks_path, 'w') as f:
            json.dump(self.chunks, f)
        print(f"Saved chunks to {chunks_path}")
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.vector_store, index_path)
        print(f"Saved FAISS index to {index_path}")
        
        # Save metadata about the embeddings
        metadata = {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_chunks": len(self.chunks),
            "embedding_dim": len(self.embeddings[0]) if self.embeddings else None
        }
        
        metadata_path = os.path.join(output_dir, "embedding_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Saved embedding metadata to {metadata_path}")

# Example usage
if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent directory (backend) and then to the data directory
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    # Define paths
    json_path = os.path.join(data_dir, "processed_incidents.json")
    output_dir = os.path.join(data_dir, "vector_store")
    
    # Create an instance of TextProcessor
    processor = TextProcessor(json_path)
    
    # Process the data
    processor.load_data()
    processor.create_chunks()
    processor.initialize_embedding_model()
    processor.generate_embeddings()
    processor.create_faiss_index()
    
    # Save the processed data
    processor.save_processed_data(output_dir)
