import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import IncidentRetriever

# Create FastAPI app
app = FastAPI(
    title="Mangalore Smart City Incident RAG API",
    description="API for querying incident data using natural language",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), "data")
vector_store_dir = os.path.join(data_dir, "vector_store")

retriever = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    num_chunks: int = 5
    model: str = "mistral"

class QueryResponse(BaseModel):
    query: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    num_chunks_retrieved: int
    processing_time_ms: float

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global retriever
    try:
        # Initialize retriever with default model
        retriever = IncidentRetriever(vector_store_dir, model_name="mistral")
        retriever.load_resources()
    except Exception as e:
        print(f"Error initializing retriever: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mangalore Smart City Incident RAG API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a natural language query about incidents
    """
    import time
    
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    # Update model if different from current
    if request.model != retriever.model_name:
        retriever.model_name = request.model
        retriever.llm = None  # Force reinitialization
    
    # Process query
    start_time = time.time()
    try:
        response = retriever.process_query(request.query, k=request.num_chunks)
        end_time = time.time()
        
        # Add processing time
        response["processing_time_ms"] = (end_time - start_time) * 1000
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        # This is a simplified version - in a real app, you'd query Ollama's API
        models = ["mistral", "llama2", "llama2:13b", "llama2:70b", "gemma:2b", "gemma:7b"]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get statistics about the incident data"""
    try:
        # Load processed data
        json_path = os.path.join(data_dir, "processed_incidents.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Calculate basic stats
        incident_types = {}
        statuses = {}
        locations = {}
        
        for record in data:
            # Count incident types
            inc_type = record.get("incident_type", "Unknown")
            incident_types[inc_type] = incident_types.get(inc_type, 0) + 1
            
            # Count statuses
            status = record.get("status", "Unknown")
            statuses[status] = statuses.get(status, 0) + 1
            
            # Count locations
            location = record.get("location", "Unknown")
            locations[location] = locations.get(location, 0) + 1
        
        return {
            "total_incidents": len(data),
            "incident_types": incident_types,
            "statuses": statuses,
            "locations": locations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
