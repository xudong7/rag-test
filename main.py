from embed import create_db, query_db, get_llm_answer
import time

def main():
  # 问题列表
  question_list = [
    "请说说文章作者的三下乡要做的事情",
    "作者大三有什么安排？",
    "总体来看，作者对于未来的规划和目标是什么？",
  ]
  
  # 创建数据库
  create_db()

  # 查询数据库并获取答案
  for question in question_list:
    print(f"\nQuerying: {question}")
    chunks = query_db(question)
    answer = get_llm_answer(question, chunks)
    print(f"Answer: {answer}\n")
    time.sleep(5)
    
if __name__ == '__main__':
  main()