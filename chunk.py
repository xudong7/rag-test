from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_data() -> str:
  """Reads the input data from a markdown file.

  Returns:
      str: The content of the markdown file.
  """
  with open("data.md", "r", encoding="utf-8") as file:
    return file.read()
  
def basic_chunking(data: str) -> list[str]:
  """Basic chunking by splitting on double newlines and preserving headers.

  Args:
      data (str): The input text data to be chunked.

  Returns:
      list[str]: A list of chunked text segments.
  """
  chunks = data.split("\n\n")

  result = []
  header = ""
  for c in chunks:
    if c.startswith("#"):
      header += f"{c}\n"
    else:
      result.append(f"{header}{c}")
      header = ""

  return result

def langchain_chunking(data: str) -> list[str]:
  """Chunking using LangChain's text splitter.

  Args:
      data (str): The input text data to be chunked.

  Returns:
      list[str]: A list of chunked text segments.
  """
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=20,           # 每个块的字符数（根据需求调整）
      chunk_overlap=5,         # 块之间的重叠字符数
      separators=["\n\n", "\n", "。", " ！", "？", "；", "……"],  # 中文优先分隔符
      keep_separator=False,
  )
  
  return splitter.split_text(data)
  
  
def get_chunks() -> list[str]:
  """Get the text chunks from the input data.

  Returns:
      list[str]: A list of text chunks.
  """
  data = read_data()
  
  # 1. just split by double newlines
  # return basic_chunking(data)

  # 2. use langchain to split by headers
  return langchain_chunking(data)

if __name__ == "__main__":
  chunks = get_chunks()
  for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")