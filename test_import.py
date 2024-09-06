import sys
import os

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

try:
    from rag import config
    print("Successfully imported app.config")
except ImportError as e:
    print("Failed to import app.config:", str(e))

try:
    from rag.config import settings
    print("Successfully imported settings from app.config")
except ImportError as e:
    print("Failed to import settings from app.config:", str(e))

print("Contents of current directory:", os.listdir())
if 'app' in os.listdir():
    print("Contents of app directory:", os.listdir('app'))