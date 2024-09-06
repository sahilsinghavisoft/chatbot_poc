import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_CONNECTION_STRING: str = ""
    OPENAI_API_KEY: str = ""
    ORGANIZATION_NAME: str = ""
    ORGANIZATION_SUMMARY: str = ""

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        env_file_encoding = 'utf-8'

settings = Settings()

# Add this for debugging
print(f"Loaded settings: {settings.dict()}")