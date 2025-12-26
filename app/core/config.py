import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "ScholarGraph AI"
    VERSION: str = "1.0.0"
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    
    # Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Collection Descriptions (for Routing)
    COLLECTIONS = {
        "cs_ai_full": "general artificial intelligence research, AI systems, AI safety, ethics",
        "ML_collection": "machine learning algorithms, theory, supervised, unsupervised, reinforcement learning",
        "dl_collection": "deep learning, neural networks, transformers, architectures, backpropagation",
        "cv_collection": "computer vision, image processing, object detection, segmentation, video analysis",
        "nlp_collection": "natural language processing, text mining, language models, translation, speech",
        "RL_collection": "reinforcement learning, robotics, control, agents, policy optimization",
        "other_cs": "computer science theory, systems, databases, security, networks, software engineering"
    }

    # Display Names for UI
    COLLECTION_DISPLAY_NAMES = {
        "cs_ai_full": "Artificial Intelligence",
        "ML_collection": "Machine Learning",
        "dl_collection": "Deep Learning",
        "cv_collection": "Computer Vision",
        "nlp_collection": "Natural Language Processing",
        "RL_collection": "Reinforcement Learning",
        "other_cs": "General Computer Science"
    }

    # Keywords for Autocomplete / Suggestions
    DOMAIN_KEYWORDS = {
        "cs_ai_full": ["artificial intelligence", "ai", "expert systems", "knowledge graph", "reasoning"],
        "ML_collection": ["machine learning", "supervised learning", "unsupervised learning", "clustering", "classification", "regression"],
        "dl_collection": ["deep learning", "neural network", "cnn", "rnn", "transformer", "lstm", "backpropagation"],
        "cv_collection": ["computer vision", "image processing", "object detection", "segmentation", "yolo", "ocr"],
        "nlp_collection": ["natural language processing", "nlp", "text mining", "bert", "gpt", "translation", "sentiment analysis"],
        "RL_collection": ["reinforcement learning", "rl", "q-learning", "policy gradient", "robotics", "agent"],
        "other_cs": ["database", "security", "network", "operating system", "software engineering"]
    }

settings = Settings()
