import openai
from rag.config import settings
from rag.models.document import TextDocument
from rag.services.embedding import EmbeddingService
import asyncio
import numpy as np
import tiktoken

openai.api_key = settings.OPENAI_API_KEY

class QAService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.organization_name = "Avisoft"
        self.organization_summary = (
            "Avisoft is a Technology and Software company based in India serving clients globally. "
            "Over years, we have delivered solutions that have helped our clients transform and scale their businesses. "
            "We offer Product Engineering, Consultancy, IT Services, Project Outsourcing, and Staff Augmentation services. "
            "With a management team having 2 decades of combined technology experience delivering complex tech platforms, "
            "we leverage our cross-functional expertise to help organizations grow their businesses. "
            "We partner with businesses to design and build Tech platforms from scratch, or to re-engineer and modernize their legacy systems."
        )
        self.owner_name = "AVinash sharma"  
        self.chatbot_purpose = (
            "This chatbot is designed to help users interact with data extracted from various documents. "
            "It can retrieve and process information from PDFs, URLs, and other sources to answer user queries. "
            "The goal is to provide insightful, relevant, and contextually accurate responses based on the extracted data."
        )

    async def get_answer(self, question: str) -> str:
        try:
            # Handle basic questions directly
            if "hi" in question.lower() or "hello" in question.lower():
                return f"Hello! How can I assist you today?"

            if "owner" in question.lower() and "avisof" in question.lower():
                return f"The owner of {self.organization_name} is {self.owner_name}."

            if "purpose" in question.lower() or "what can you do" in question.lower():
                return f"{self.chatbot_purpose}"

            # Generate the embedding for the question
            question_embedding = await self.embedding_service.generate_embedding(question)

            # Perform vector search to find similar documents
            similar_docs = await self.vector_search(question_embedding)

            if not similar_docs:
                return "I couldn't find any relevant information in the database to answer your question."

            # Truncate the context to fit within token limits
            context = self.truncate_context(similar_docs, question)

            system_message = f"""Hello! You are interacting with a service developed by {self.organization_name}. 

You are a helpful AI assistant representing {self.organization_name}. 
Your task is to answer the user's query based on the provided context. 
Follow these guidelines:
1. Use only the information from the given context to answer the query.
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
3. Cite the relevance scores when referring to specific pieces of information.
4. Provide a concise yet informative answer.

Organization Summary: {self.organization_summary}

Special Instructions:
- If the user asks who developed this service, respond with: "This service was developed by {self.organization_name}."
- If the user asks about the owner of the organization, respond with: "The owner of {self.organization_name} is {self.owner_name}."
- If the user asks about the purpose of the chatbot, respond with: "{self.chatbot_purpose}"
- For greetings like "hi" or "hello," respond with: "Hello! How can I assist you today?"
- For other questions, use the provided context and guidelines to generate a response based on the content and relevance scores.
"""

            # Get the response from OpenAI
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0.8
            )
            answer = response['choices'][0]['message']['content']
            return answer

        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"

    async def vector_search(self, query_embedding, k=5):
        documents = TextDocument.objects.all()
        similarities = []

        for doc in documents:
            if doc.embedding:  # Check if embedding exists
                doc_embedding = np.array(doc.embedding)
                query_embedding_np = np.array(query_embedding)
                similarity = np.dot(doc_embedding, query_embedding_np) / (np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding_np))
                similarities.append((doc, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        results = [
            {
                'content': doc.content,
                'score': float(similarity),
                'source_url': getattr(doc, 'source_url', None) 
            }
            for doc, similarity in top_k
        ]

        return results

    def truncate_context(self, similar_docs, question, max_tokens=4096):
        reserved_tokens = 500
        available_tokens = max_tokens - reserved_tokens
        context_parts = []
        current_tokens = 0

        for doc in similar_docs:
            content = f"Content (Score: {doc['score']:.2f})"
            if doc['source_url']:
                content += f" (Source: {doc['source_url']})"
            content += f": {doc['content']}"
            
            content_tokens = self.tokenizer.encode(content)
            
            if current_tokens + len(content_tokens) > available_tokens:
                remaining_tokens = available_tokens - current_tokens
                truncated_content = self.tokenizer.decode(content_tokens[:remaining_tokens])
                context_parts.append(truncated_content)
                break
            
            context_parts.append(content)
            current_tokens += len(content_tokens)

        return "\n".join(context_parts)
