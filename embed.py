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
  """Generate embeddings for the given text.

  Args:
      text (str): The text to embed.
      store (bool): Whether to store the embedding.
      retry_count (int, optional): The number of retry attempts. Defaults to 3.

  Returns:
      list[float]: The generated embedding.
  """
  for attempt in range(retry_count):
    result = google_client.models.embed_content(
      model=EMBEDDING_MODEL,
      contents=text,
      config={
        "task_type": "RETRIEVAL_DOCUMENT" if store else "RETRIEVAL_QUERY"
      }
    )

    time.sleep(5)

    assert result.embeddings
    assert result.embeddings[0].values
    return result.embeddings[0].values

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
    time.sleep(5)
    
  print("Database creation completed!")
  time.sleep(5)

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

if __name__ == '__main__':
  # 问题列表
  question_list = [
    "请说说文章作者的三下乡要做的事情",
    "作者大三有什么安排？",
    "总体来看，作者对于未来的规划和目标是什么？",
  ]
  
  # 创建数据库
  # create_db()

  # 查询数据库并获取答案
  for question in question_list:
    print(f"\nQuerying: {question}")
    chunks = query_db(question)
    answer = get_llm_answer(question, chunks)
    print(f"Answer: {answer}\n")
    time.sleep(5)