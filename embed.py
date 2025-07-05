import chunk
import chromadb
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# 加载环境变量
load_dotenv()

# 配置Google API key
API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
LLM_MODEL = "gemini-2.5-flash-preview-05-20"
google_client = genai.Client(api_key=API_KEY)

chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection("rag_test_collection")

def embed(text: str, store: bool, retry_count: int = 3) -> list[float]:
  for attempt in range(retry_count):
    result = google_client.models.embed_content(
      model=EMBEDDING_MODEL,
      contents=text,
      config={
        "task_type": "RETRIEVAL_DOCUMENT" if store else "RETRIEVAL_QUERY"
      }
    )

    time.sleep(2)

    assert result.embeddings
    assert result.embeddings[0].values
    return result.embeddings[0].values

def create_db() -> None:
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
    time.sleep(2)

def query_db(question: str) -> list[str]:
  question_embedding = embed(question, store=False)
  result = chromadb_collection.query(
    query_embeddings=question_embedding,
    n_results=5
  )
  assert result["documents"]
  return result["documents"][0]


if __name__ == '__main__':
  question = "请说说文章作者的三下乡要做的事情"
  
  print("Creating database...")
  create_db()
  print("Database creation completed!")
  time.sleep(5)
  
  print(f"\nQuerying: {question}")
  chunks = query_db(question)
  
  prompt = "Please answer user's question according to context\n"
  prompt += f"Question: {question}\n"
  prompt += "Context:\n"
  for c in chunks:
    prompt += f"{c}\n"
    prompt += "-------------\n"
  
  print("Generating response...")
  result = google_client.models.generate_content(
    model=LLM_MODEL,
    contents=prompt
  )
  print("\nResponse:")
  print(result.text if hasattr(result, 'text') else result)