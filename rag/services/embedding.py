from sentence_transformers import SentenceTransformer
from rag.models.document import TextDocument
import asyncio
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def generate_embedding(self, text: str) -> list:
        embedding = await asyncio.to_thread(self.model.encode, text)
        return embedding.tolist()  # Convert numpy array to list of floats

    async def update_document_embedding(self, doc_id: str):
        doc = await asyncio.to_thread(TextDocument.objects(id=doc_id).first)
        if doc:
            doc.embedding = await self.generate_embedding(doc.content)
            await asyncio.to_thread(doc.save)
        else:
            raise ValueError(f"No document found with id: {doc_id}")