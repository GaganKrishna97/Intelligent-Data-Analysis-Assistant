import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    HF_API_KEY = os.getenv("HF_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))
