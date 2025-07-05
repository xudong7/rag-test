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
  """Generate embeddings using the Gemini model.

  Args:
      text (str): The text to embed.
      store (bool): Whether to store the embedding.

  Returns:
      list[float]: The generated embedding.
  """
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
  """Generate embeddings for the given text.

  Args:
      text (str): The text to embed.
      store (bool): Whether to store the embedding.
      retry_count (int, optional): The number of retry attempts. Defaults to 3.

  Returns:
      list[float]: The generated embedding.
  """
  
  # 1. use Gemini embedding
  # return gemini_embed(text, store)
  
  # 2. use SentenceTransformer embedding
  return embed_model.encode(text, normalize_embeddings=True).tolist()

def create_db() -> None:
  """Create the database and populate it with document embeddings.
  """
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

def query_db(question: str) -> list[str]:
  """Query the database for relevant documents based on the question.

  Args:
      question (str): The question to query.

  Returns:
      list[str]: A list of relevant documents.
  """
  question_embedding = embed(question, store=False)
  result = chromadb_collection.query(
    query_embeddings=question_embedding,
    n_results=5
  )
  assert result["documents"]
  return result["documents"][0]

def get_llm_answer(question: str, context: list[str]) -> str:
  """Get the answer from the LLM based on the question and context.

  Args:
      question (str): The question to ask.
      context (list[str]): The context to provide to the LLM.

  Returns:
      str: The answer from the LLM.
  """
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

if __name__ == "__main__":
  print(embed("你好，世界！", store=True))    