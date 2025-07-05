def read_data() -> str:
  with open("data.md", "r", encoding="utf-8") as file:
    return file.read()
  
def get_chunks() -> list[str]:
  data = read_data()
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

if __name__ == "__main__":
  chunks = get_chunks()
  for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")