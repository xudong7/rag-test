import chunk
import chromadb
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from sentence_transformers import SentenceTransformer

# 加载环境变量
load_dotenv()

# Google API key
API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
LLM_MODEL = "gemini-2.5-flash-preview-05-20"

# Embedding models
google_client = genai.Client(api_key=API_KEY)
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# Vector database
chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection("rag_test_collection")

def gemini_embed(text: str, store: bool) -> list[float]:
  result = google_client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=text,
    config={
      "task_type": "RETRIEVAL_DOCUMENT" if store else "RETRIEVAL_QUERY"
    }
  )
  time.sleep(5)
  return result.embeddings[0].values

def embed(text: str, store: bool) -> list[float]:
  # 1. use Gemini embedding
  # return gemini_embed(text, store)
  
  # 2. use SentenceTransformer embedding
  return embed_model.encode(text, normalize_embeddings=True).tolist()

def create_db() -> None:
  print("Creating database...")
  chunks = chunk.get_chunks()
  total_chunks = len(chunks)
  
  for idx, c in enumerate(chunks):
    embedding = embed(c, store=True)
    chromadb_collection.upsert(
      ids=str(idx),
      documents=c,
      embeddings=embedding
    )
    print(f"Successfully processed chunk {idx + 1}")
    
  print("Database creation completed!")

def create_db_from_multiple_docs(file_patterns: list[str] = None, source_dir: str = "data_source") -> None:
  print(f"Creating database from multiple documents in '{source_dir}' directory...")
  chunk_infos = chunk.get_chunks_from_multiple_docs(file_patterns, source_dir)
  total_chunks = len(chunk_infos)
  
  for idx, chunk_info in enumerate(chunk_infos):
    text = chunk_info["text"]
    source = chunk_info["source"]
    metadata = chunk_info["metadata"]
    
    embedding = embed(text, store=True)
    chromadb_collection.upsert(
      ids=str(idx),
      documents=text,
      embeddings=embedding,
      metadatas=metadata
    )
    print(f"Successfully processed chunk {idx + 1}/{total_chunks} from {source}")
    
  print("Database creation completed!")

def query_db(question: str) -> list[str]:
  question_embedding = embed(question, store=False)
  result = chromadb_collection.query(
    query_embeddings=question_embedding,
    n_results=5
  )
  assert result["documents"]
  return result["documents"][0]

def query_db_with_metadata(question: str) -> tuple[list[str], list[dict]]:
  question_embedding = embed(question, store=False)
  result = chromadb_collection.query(
    query_embeddings=question_embedding,
    n_results=5,
    include=["documents", "metadatas"]
  )
  assert result["documents"]
  documents = result["documents"][0]
  metadatas = result["metadatas"][0] if result["metadatas"] else [{}] * len(documents)
  return documents, metadatas

def get_llm_answer(question: str, context: list[str]) -> str:
  prompt = "Please answer user's question according to context\n"
  prompt += f"Question: {question}\n"
  prompt += "Context:\n"
  for c in context:
    prompt += f"{c}\n"
    prompt += "-------------\n"
  
  print("Generating response...")
  result = google_client.models.generate_content(
    model=LLM_MODEL,
    contents=prompt
  )
  
  return result.text if hasattr(result, 'text') else result

def get_llm_answer_with_sources(question: str, context: list[str], metadatas: list[dict]) -> str:
  prompt = "Please answer user's question according to context. Include source information when relevant.\n"
  prompt += f"Question: {question}\n"
  prompt += "Context:\n"
  for i, c in enumerate(context):
    source_info = metadatas[i].get('file_name', 'Unknown') if i < len(metadatas) else 'Unknown'
    prompt += f"[Source: {source_info}] {c}\n"
    prompt += "-------------\n"
  
  print("Generating response...")
  result = google_client.models.generate_content(
    model=LLM_MODEL,
    contents=prompt
  )
  
  return result.text if hasattr(result, 'text') else result

if __name__ == "__main__":
  print(embed("你好，世界！", store=True))    