import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for API keys and model settings"""
    
    # API Keys
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Default Models
    # Text extraction/summarization model (Hugging Face)
    TEXT_EXTRACTION_MODEL = "facebook/bart-large-cnn"
    
    # OpenAI model for text generation (keywords and questions)
    OPENAI_MODEL = "gpt-3.5-turbo"
    # Alternative: "gpt-4" for better quality (more expensive)
    
    # Alternative models you can use:
    # For extraction: "google/pegasus-xsum", "t5-base"
    # For generation: "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"
    
    # API Settings
    API_TIMEOUT = 60
    MAX_RETRIES = 3
    
    @classmethod
    def validate(cls):
        """Validate that required API tokens are set"""
        if not cls.HUGGINGFACE_API_TOKEN:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables. Get one at: https://huggingface.co/settings/tokens")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Get one at: https://platform.openai.com/api-keys")
        return True