import json
import sys
import os
import pickle
import string
import re
import networkx as nx
import string
from collections import defaultdict


from data_loader import PlainDataLoader


def build_graph(class_name):
    """
    构建一个字的有向图，边的权重为字对在语料库中出现的频率
    """
    loader = PlainDataLoader()
    if class_name == "shi":
        corpus = loader.extract_from_multiple(["tangsong","yudingquantangshi","shuimotangshi"])
    else:
        corpus = loader.body_extractor(class_name)

    G = nx.DiGraph()
    edge_weights = defaultdict(int)

    # 用滑动窗口构建图（以字为单位）
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            u = sentence[i]
            v = sentence[i + 1]
            edge_weights[(u, v)] += 1

    # 将边加入图中
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    with open(f"graph_{class_name}.pkl", "wb") as f:
        pickle.dump(G, f)
    
    # 加权 PageRank，考虑出现频率
    pr = nx.pagerank(G, weight='weight')

    with open(f"word_{class_name}.pkl", "wb") as f:
        pickle.dump(pr, f)

def load_graph(class_name):
    with open(f"./word_data/word_{class_name}.pkl", "rb") as f:
        pr = pickle.load(f)
    with open(f"./word_data/graph_{class_name}.pkl", "rb") as f:
        G = pickle.load(f)   
    return pr, G

def load_word_set(filepath):
    """
    从 txt 文件中加载词语集合，按空格分隔，每行多个词
    返回 set 类型 word_set
    """
    word_set = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                if word:
                    word_set.add(word)
    return word_set

def recommend_next_char(prefix, pr, G, word_set=None, top_k=10):
    """
    根据前缀 prefix 预测下一个字，并尝试组成词
    参数：
        prefix: 输入的前缀字符串
        pr: 字的概率字典（用于排序）
        G: 字图，每个字连接到可能的下一个字集合
        word_set: （可选）词典集合，判断是否能组成真实词语
        top_k: 返回数量上限
    返回：
        推荐字列表；如词典可用则同时输出组成词
    """ 
    if not prefix:
        return "😢(还没输入)"

    last_char = prefix[-1]
    if last_char not in G:
        return "😢(适合结束这句话了)"

    chinese_punctuation = "，。、！？【】（）《》“”‘’：；——…·"
    all_punctuation = set(string.punctuation + chinese_punctuation)

    # 候选字：从 last_char 出发的可能字
    candidates = G[last_char]
    ranked = sorted(
        [(c, pr.get(c, 0)) for c in candidates],
        key=lambda x: -x[1]
    )

    # 过滤掉标点，仅取 top_k 个字
    filtered = [char for char, _ in ranked if char not in all_punctuation]
    top_chars = filtered[:top_k]

    # 返回字推荐
    char_result = "🔡: " + "、".join(top_chars) if top_chars else "😢(适合结束这句话了)"

    # 若给定词典，尝试组成词
    if word_set:
        word_suggestions = []
        for ch in top_chars: 
            i = 0
            for word in word_set:
                if ch == word[0]:
                    word_suggestions.append(word)
                    i+=1
                if i > 5:
                    break
        word_result = "\n\n 可组词语：" + "、".join(word_suggestions) if word_suggestions else "\n"
    else:
        word_result = ""

    return char_result + word_result


if __name__ == "__main__":
    class_name = "chuci" 
    # build_graph(class_name)
    
    prefix = "春天的花"
    word_set = load_word_set("./data/songci/fenci_shi.txt")
    
    pr, G = load_graph(class_name)
    next_chars = recommend_next_char(prefix, pr, G, word_set=word_set ,top_k=10)
    print(f"前缀 '{prefix}' 的下一个字预测：{next_chars}")