import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def load_config():
    """Load configuration values from environment variables."""
    config = {
        # Model keys
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),

        # Qdrant vector DB
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sales_counsellor"),

        # Lead CRM file (PoC)
        "CRM_EXCEL_PATH": os.getenv("CRM_EXCEL_PATH", "data/leads_crm.xlsx"),

        # Signal log file (PoC)
        "SIGNAL_LOG_PATH": os.getenv("SIGNAL_LOG_PATH", "data/signals_log.json"),

        # Model selection
        "USE_MODEL": os.getenv("USE_MODEL", "openai"),  # or "gemini"
    }

    # Validate required keys
    required = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return config

# Quick access
config = load_config()
GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]
QDRANT_URL = config["QDRANT_URL"]
QDRANT_API_KEY = config["QDRANT_API_KEY"]
COLLECTION_NAME = config["QDRANT_COLLECTION"]
CRM_EXCEL_PATH = config["CRM_EXCEL_PATH"]
SIGNAL_LOG_PATH = config["SIGNAL_LOG_PATH"]
USE_MODEL = config["USE_MODEL"]
