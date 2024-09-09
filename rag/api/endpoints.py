from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl
from rag.services.data_capture import DataCaptureService
from rag.services.embedding import EmbeddingService
from rag.services.qa import QAService
from config import settings
import logging
import os

router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

class URLInput(BaseModel):
    url: HttpUrl

class QuestionInput(BaseModel):
    question: str

# Use the diffbot_token from settings
@router.post("/capture/url")
async def capture_url(url_input: URLInput):
    try:
        data_capture_service = DataCaptureService(diffbot_token=settings.diffbot_token,openai_api_key=settings.OPENAI_API_KEY)
        doc_id = await data_capture_service.capture_url(str(url_input.url))
        embedding_service = EmbeddingService()
        await embedding_service.update_document_embedding(doc_id)
        return {"message": "URL captured and processed successfully", "document_id": doc_id}
    except Exception as e:
        logger.error(f"Error capturing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/capture/pdf")
async def capture_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        data_capture_service = DataCaptureService(diffbot_token=settings.diffbot_token,openai_api_key=settings.OPENAI_API_KEY)
        doc_id = await data_capture_service.capture_pdf(file_location)

        embedding_service = EmbeddingService()
        await embedding_service.update_document_embedding(doc_id)
        return {"message": "PDF captured and processed successfully", "document_id": doc_id}
    except Exception as e:
        logger.error(f"Error capturing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))  # Fixed missing parenthesis
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@router.post("/qa")
async def answer_question(question_input: QuestionInput):
    try:
        qa_service = QAService()
        answer = await qa_service.get_answer(question_input.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
