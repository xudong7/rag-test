from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob

def read_data() -> str:
  with open("data.md", "r", encoding="utf-8") as file:
    return file.read()

def read_multiple_documents(file_patterns: list[str] = None, source_dir: str = "data_source") -> dict[str, str]:
  if file_patterns is None:
    file_patterns = ["*.md", "*.txt", "*.doc", "*.docx"]
  
  documents = {}
  
  if not os.path.exists(source_dir):
    print(f"Warning: Source directory '{source_dir}' does not exist")
    return documents
  
  for pattern in file_patterns:
    # 在指定目录中搜索文件
    search_pattern = os.path.join(source_dir, pattern)
    files = glob.glob(search_pattern)
    for file_path in files:
      try:
        # 根据文件扩展名选择合适的读取方式
        if file_path.endswith(('.md', '.txt')):
          with open(file_path, "r", encoding="utf-8") as file:
            documents[file_path] = file.read()
        elif file_path.endswith('.docx'):
          try:
            from docx import Document
            doc = Document(file_path)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            documents[file_path] = content
          except ImportError:
            print(f"Warning: python-docx not installed, skipping {file_path}")
        elif file_path.endswith('.doc'):
          print(f"Warning: .doc files not supported, skipping {file_path}")
      except Exception as e:
        print(f"Error reading {file_path}: {e}")
  
  return documents
  
def basic_chunking(data: str) -> list[str]:
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
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=200,           # 每个块的字符数（根据需求调整）
      chunk_overlap=50,         # 块之间的重叠字符数
      separators=["\n\n", "\n", "。", " ！", "？", "；", "……"],  # 中文优先分隔符
      keep_separator=False,
  )
  
  return splitter.split_text(data)
  
def get_chunks() -> list[str]:
  data = read_data()
  
  # 1. just split by double newlines
  # return basic_chunking(data)

  # 2. use langchain to split by headers
  return langchain_chunking(data)

def get_chunks_from_multiple_docs(file_patterns: list[str] = None, source_dir: str = "data_source") -> list[dict]:
  documents = read_multiple_documents(file_patterns, source_dir)
  all_chunks = []
  
  if not documents:
    print(f"No documents found in '{source_dir}' directory with patterns: {file_patterns}")
    return all_chunks
  
  print(f"Found {len(documents)} documents in '{source_dir}' directory")
  
  for file_path, content in documents.items():
    chunks = langchain_chunking(content)
    print(f"Processing {file_path}: {len(chunks)} chunks")
    for chunk in chunks:
      all_chunks.append({
        "text": chunk,
        "source": file_path,
        "metadata": {
          "file_name": os.path.basename(file_path),
          "file_path": file_path,
          "source_directory": source_dir
        }
      })
  
  return all_chunks

if __name__ == "__main__":
  # 测试单文档处理
  print("=== 单文档处理 ===")
  chunks = get_chunks()
  for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
  
  # 测试多文档处理
  print("\n=== 多文档处理 ===")
  multi_chunks = get_chunks_from_multiple_docs()
  for i, chunk_info in enumerate(multi_chunks):
    print(f"Chunk {i+1} (from {chunk_info['source']}):\n{chunk_info['text']}\n")