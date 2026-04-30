# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# class Config:
#     """
#     Configuration class to manage environment variables.
#     All hyperparameters and paths are loaded from .env.
#     """
#     # Neo4j Settings
#     NEO4J_URI = os.getenv("NEO4J_URI")
#     NEO4J_USERNAME = os.getenv("NEO4J_USER")
#     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

#     # Ollama Settings
#     OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
#     OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

#     # Embedding Settings - Defaulting to the official HF hub to fix the pooling error
#     EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "mixedbread-ai/mxbai-embed-large-v1")
#     EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

#     # Hyperparameters
#     VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
#     VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1024))
#     SIMILARITY_FUNCTION = os.getenv("SIMILARITY_FUNCTION", "cosine")

#     # Data Path
#     CLEANED_RECORDS_PATH = os.path.join(os.getenv("CLEANED_OUTPUT_FOLDER", "."), "cleaned_records.json")
    