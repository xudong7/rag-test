from embed import create_db_from_multiple_docs, query_db_with_metadata, get_llm_answer_with_sources
import time

class Config:
  FILE_PATTERNS = ["*.md", "*.txt"]  # 可以添加更多格式，如 "*.docx"
  SOURCE_DIR = "data_source"  # 指定源文件目录

def main_multi_docs():
  question_list = [
    "请说说文章作者的三下乡要做的事情",
    "作者大三有什么安排？",
    "总体来看，作者对于未来的规划和目标是什么？",
    "目前已完成的项目有哪些？",
  ]
  
  # 从多个文档创建数据库
  create_db_from_multiple_docs(Config.FILE_PATTERNS, Config.SOURCE_DIR)

  for question in question_list:
    print(f"\n问题: {question}")
    print("-" * 50)
    
    # 获取相关文档和元数据
    chunks, metadatas = query_db_with_metadata(question)
    
    # 显示找到的相关文档来源
    sources = set()
    for metadata in metadatas:
        if metadata and 'file_name' in metadata:
            sources.add(metadata['file_name'])
    
    if sources:
        print(f"相关文档来源: {', '.join(sources)}")
    
    # 获取带来源信息的回答
    answer = get_llm_answer_with_sources(question, chunks, metadatas)
    print(f"回答: {answer}")
    
    time.sleep(2)

if __name__ == '__main__':    
    # 运行多文档RAG系统
    main_multi_docs()
