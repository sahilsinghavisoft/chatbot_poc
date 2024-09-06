import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fastapi import FastAPI
from mongoengine import connect,disconnect
import os
import uvicorn
from dotenv import load_dotenv
from rag.config import settings# Ensure this import works correctly
from rag.api.endpoints import router  

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="RAG System API")


@app.on_event("startup")
def startup_db_client():
    # Disconnect any existing connections with the alias 'default'
    disconnect(alias='default')
    
    # Establish a new connection with a specific alias if needed
    connect(
        db='your_database_name',
        host=settings.MONGO_CONNECTION_STRING,
        alias='default'  # Use an alias to differentiate connections
    )

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("rag.main:app", host="0.0.0.0", port=8000, reload=True)