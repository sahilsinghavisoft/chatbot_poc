from goose3 import Goose
from langchain_community.document_loaders import PyPDFLoader
from rag.models.document import TextDocument
from rag.services.embedding import EmbeddingService
from newspaper import Article
from bs4 import BeautifulSoup
import requests
import asyncio
from typing import Tuple, Optional
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class DataCaptureService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.logger = logging.getLogger(__name__)

    async def capture_url(self, url: str) -> str:
        try:
            extractors = [
                self._extract_with_beautiful_soup,
                self._try_newspaper,
                self._try_goose,
                self._extract_with_selenium
            ]
            max_content, max_metadata = "", {}

            for extractor in extractors:
                try:
                    content, metadata = await extractor(url)
                    if content and len(content) > len(max_content):
                        max_content, max_metadata = content, metadata
                except Exception as e:
                    self.logger.warning(f"Extractor {extractor.__name__} failed for URL {url}: {str(e)}")

            if not max_content:
                raise ValueError("Failed to extract content from the URL")

            return await self._process_and_save_content(max_content, url, max_metadata)
        except Exception as e:
            self.logger.error(f"Error capturing URL {url}: {str(e)}")
            raise

    async def capture_pdf(self, file_path: str) -> str:
        try:
            content = await self._extract_pdf_content(file_path)
            return await self._process_and_save_content(content, file_path)
        except Exception as e:
            self.logger.error(f"Error capturing PDF {file_path}: {str(e)}")
            raise

    async def _extract_with_beautiful_soup(self, url: str) -> Tuple[Optional[str], Optional[dict]]:
        try:
            response = await asyncio.to_thread(requests.get, url)
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text(separator="\n").strip()
            metadata = {
                "source": "BeautifulSoup",
                "title": soup.title.string if soup.title else "No Title",
                "authors": ["N/A"],
                "publish_date": "N/A"
            }
            return content, metadata
        except Exception as e:
            self.logger.warning(f"BeautifulSoup extraction failed for URL {url}: {str(e)}")
            return None, None

    async def _try_newspaper(self, url: str) -> Tuple[Optional[str], Optional[dict]]:
        try:
            article = Article(url)
            await asyncio.to_thread(article.download)
            await asyncio.to_thread(article.parse)
            if article.text:
                return article.text, {
                    "source": "Newspaper3k",
                    "title": article.title,
                    "authors": article.authors,
                    "publish_date": article.publish_date
                }
            return None, None
        except Exception as e:
            self.logger.warning(f"Newspaper extraction failed for URL {url}: {str(e)}")
            return None, None

    async def _try_goose(self, url: str) -> Tuple[Optional[str], Optional[dict]]:
        try:
            g = Goose()
            article = await asyncio.to_thread(g.extract, url=url)
            if article.cleaned_text:
                return article.cleaned_text, {
                    "source": "Goose3",
                    "title": article.title,
                    "authors": article.authors,
                    "publish_date": article.publish_date
                }
            return None, None
        except Exception as e:
            self.logger.warning(f"Goose extraction failed for URL {url}: {str(e)}")
            return None, None

    async def _extract_with_selenium(self, url: str) -> Tuple[Optional[str], Optional[dict]]:
        try:
            options = Options()
            options.add_argument("--headless")
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.get(url)
            content = driver.find_element("tag name", "body").text
            title = driver.title
            metadata = {
                "source": "Selenium",
                "title": title,
                "authors": ["N/A"],
                "publish_date": "N/A"
            }
            driver.quit()
            return content, metadata
        except Exception as e:
            self.logger.warning(f"Selenium extraction failed for URL {url}: {str(e)}")
            return None, None

    async def _extract_pdf_content(self, file_path: str) -> str:
        loader = PyPDFLoader(file_path)
        pages = await asyncio.to_thread(loader.load_and_split)
        return "\n".join([page.page_content for page in pages])

    async def _process_and_save_content(self, content: str, source_url: str, metadata: dict = None) -> str:
        document_embedding = await self.embedding_service.generate_embedding(content)
        doc = TextDocument(content=content, source_url=source_url, embedding=document_embedding)
        await asyncio.to_thread(doc.save)
        self._log_capture_info(metadata, content)
        return str(doc.id)

    def _log_capture_info(self, metadata: dict, content: str):
        if metadata:
            self.logger.info(f"Source: {metadata.get('source', 'Unknown')}")
            self.logger.info(f"Title: {metadata.get('title', 'No Title')}")
            self.logger.info(f"Authors: {', '.join(metadata.get('authors', ['N/A']))}")
            self.logger.info(f"Publish Date: {metadata.get('publish_date', 'N/A')}")
        self.logger.info(f"Content preview: {content[:500]}...")
