
import asyncio
import logging
import requests
from typing import Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader
from rag.models.document import TextDocument
from rag.services.embedding import EmbeddingService
from config import settings  # Importing settings from config.py
import openai
from newspaper import Article as NewspaperArticle
from goose3 import Goose
from bs4 import BeautifulSoup

class DataCaptureService:
    def __init__(self, diffbot_token: str, openai_api_key: str):
        self.embedding_service = EmbeddingService()
        self.logger = logging.getLogger(__name__)
        self.diffbot_token = diffbot_token
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    async def capture_url(self, url: str) -> str:
        try:
            content, metadata = await self._extract_with_diffbot(url)
            if not content:
                self.logger.info("Diffbot failed, attempting alternative extraction methods.")
                content, metadata = await self._extract_with_alternatives(url)
                if not content:
                    raise ValueError("Failed to extract content from the URL using all methods")

            enriched_content = await self._enhance_content_with_gpt(content)
            return await self._process_and_save_content(enriched_content, url, metadata)
        except Exception as e:
            self.logger.error(f"Error capturing URL {url}: {str(e)}")
            raise

    async def capture_pdf(self, file_path: str) -> str:
        try:
            content = await self._extract_pdf_content(file_path)
            enriched_content = await self._enhance_content_with_gpt(content)
            return await self._process_and_save_content(enriched_content, file_path)
        except Exception as e:
            self.logger.error(f"Error capturing PDF {file_path}: {str(e)}")
            raise

    async def _extract_with_diffbot(self, url: str) -> Tuple[Optional[str], Optional[dict]]:
        try:
            api_url = f"https://api.diffbot.com/v3/article?token={self.diffbot_token}&url={url}"
            response = await asyncio.to_thread(requests.get, api_url)
            if response.status_code == 200:
                data = response.json()
                article = data.get('objects', [{}])[0]
                content = article.get('text', '').strip()
                metadata = {
                    "source": "Diffbot",
                    "title": article.get('title', 'No Title'),
                    "authors": article.get('author', ['N/A']),
                    "publish_date": article.get('date', 'N/A')
                }
                return content, metadata
            else:
                self.logger.warning(f"Diffbot API failed with status code {response.status_code}")
                return None, None
        except Exception as e:
            self.logger.warning(f"Diffbot extraction failed for URL {url}: {str(e)}")
            return None, None

    async def _extract_with_alternatives(self, url: str) -> Tuple[Optional[str], Optional[dict]]:
        try:
            content, metadata = await asyncio.to_thread(self._extract_with_newspaper, url)
            if content:
                return content, metadata
        except Exception as e:
            self.logger.warning(f"Newspaper3k extraction failed for URL {url}: {str(e)}")

        try:
            content, metadata = await asyncio.to_thread(self._extract_with_goose, url)
            if content:
                return content, metadata
        except Exception as e:
            self.logger.warning(f"Goose extraction failed for URL {url}: {str(e)}")

        try:
            content, metadata = await asyncio.to_thread(self._extract_with_beautifulsoup, url)
            if content:
                return content, metadata
        except Exception as e:
            self.logger.warning(f"BeautifulSoup extraction failed for URL {url}: {str(e)}")

        return None, None

    def _extract_with_newspaper(self, url: str) -> Tuple[Optional[str], dict]:
        article = NewspaperArticle(url)
        article.download()
        article.parse()
        return article.text, {
            "source": "Newspaper3k",
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date
        }

    def _extract_with_goose(self, url: str) -> Tuple[Optional[str], dict]:
        goose = Goose()
        article = goose.extract(url=url)
        return article.cleaned_text, {
            "source": "Goose",
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date
        }

    def _extract_with_beautifulsoup(self, url: str) -> Tuple[Optional[str], dict]:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text(separator='\n').strip()
            title = soup.title.string if soup.title else 'No Title'
            return content, {
                "source": "BeautifulSoup",
                "title": title,
                "authors": ['N/A'],
                "publish_date": 'N/A'
            }
        else:
            self.logger.warning(f"BeautifulSoup extraction failed for URL {url}: Status code {response.status_code}")
            return None, None

    async def _extract_pdf_content(self, file_path: str) -> str:
        loader = PyPDFLoader(file_path)
        pages = await asyncio.to_thread(loader.load_and_split)
        return "\n".join([page.page_content for page in pages])

    async def _enhance_content_with_gpt(self, content: str) -> str:
        try:
            response = await asyncio.to_thread(
                openai.Completion.create,
                model="gpt-4o-mini",
                prompt=(
                "Please enhance the following content. Focus on improving clarity, readability, and engagement while maintaining the original meaning and context. "
                "Add any additional details or explanations that could help the reader better understand the content. If applicable, make the language more concise and impactful.\n\n"
                f"Content:\n{content}"
            ),
                max_tokens=4000 
            )
            return response.choices[0].text.strip()
        except Exception as e:
            self.logger.warning(f"GPT-4 enhancement failed: {str(e)}")
            return content

    async def _process_and_save_content(self, content: str, source_url: str, metadata: dict = None) -> str:
        document_embedding = await self.embedding_service.generate_embedding(content)
        doc = TextDocument(content=content, source_url=source_url, embedding=document_embedding)
        await asyncio.to_thread(doc.save)
        self._log_capture_info(metadata, content)
        return str(doc.id)

    def _log_capture_info(self, metadata: dict, content: str):
        if metadata:
            self.logger.info(f"Source: {metadata.get('source', 'Diffbot')}")
            self.logger.info(f"Title: {metadata.get('title', 'No Title')}")
            self.logger.info(f"Authors: {', '.join(metadata.get('authors', ['N/A']))}")
            self.logger.info(f"Publish Date: {metadata.get('publish_date', 'N/A')}")
        self.logger.info(f"Content preview: {content[:500]}...")
# Initialize the DataCaptureService with the token from the settings
data_capture_service = DataCaptureService(diffbot_token=settings.diffbot_token, openai_api_key=settings.OPENAI_API_KEY)
