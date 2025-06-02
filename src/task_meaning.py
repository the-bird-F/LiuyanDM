import json
import sys
import os
import jieba
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from data_loader import PlainDataLoader


def clean_line(line):
    """清洗一行文本，去除标点符号和非中文字符，并分句"""
    # 确定分句标签（，保留）
    line = re.sub(r'[。、；！？：.!?;:「」‘’“”()（）《》【】[\]{}<>——\-~·…]', '|', line) 
    line = re.sub(r'[^\u4e00-\u9fa5\|]', '', line)
    line_list = line.split('|')
    line_list = [part.strip() for part in line_list if part.strip()]
    return line_list


def build_vectorstore(load_flag: bool, class_name: str , embed_model) -> Chroma:
    """
    构建一个向量数据库检索器
    """
    if load_flag:
        # 如果已经存在数据库，则加载它
        vectorstore = Chroma(
            persist_directory=f"./chroma_data/{class_name}",
            embedding_function=embed_model
        )
    else:
        # 如果不存在数据库，则创建它
        # Load sentence 
        loader = PlainDataLoader()
        if class_name == "shi":
            data = loader.body_extractor("yudingquantangshi")
        else:
            data = loader.body_extractor(class_name)
        
        sentences = set()  # 使用集合去重
        for _, line in enumerate(data):
            cleaned_line = clean_line(line)
            sentences.update(cleaned_line)  
        sentences = list(sentences)

        print(f"总句子数：{len(sentences)}")
        print(f"前5句子：{sentences[:5]}")
        
        # 构建数据库（自动持久化）
        vectorstore = Chroma.from_texts(
            texts=sentences,        
            embedding=embed_model,
            persist_directory=f"chroma_data/{class_name}",
        )
        
    return vectorstore



def recommend_sentences(input: str, vectorstore: Chroma, top_n: int = 5, threshold: float = 0.1) -> list:
    """
    使用向量数据库推荐句子
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":top_n})
    sen = set()
    sen_list = retriever.invoke(input)
    for sentence in sen_list:
        sen.add(sentence.page_content)
    return "\n".join(sen)

if __name__ == "__main__":
    # 示例：构建向量数据库并进行检索
    class_name = "shi"  
    load_flag = False  # 是否加载已存在的数据库
    
    embed_model = HuggingFaceEmbeddings(
                model_name=f'maidalun1020/bce-embedding-base_v1',
                # model_name=f'../../bce-embedding-base_v1',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 8, 'normalize_embeddings': False}
            )
    
    vectorstore = build_vectorstore(load_flag, class_name, embed_model)
    input_text = "春天的花开得很美" # 输入文本
    recommended_sentences = recommend_sentences(input_text, vectorstore, top_n=5)
    print("推荐的句子：")
    print(recommended_sentences)