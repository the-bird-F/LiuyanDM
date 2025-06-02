import pandas as pd
import ast
import os
import jieba

# 加载频繁项集数据
def load_data(style):
    data_dir =  f'./data/{style}'
    
    itemsets_path = os.path.join(data_dir, 'merged_frequent_itemsets.json')
    itemsets_df = pd.read_json(itemsets_path)

    # 如果是字符串格式，转换成列表
    if isinstance(itemsets_df.loc[0, 'itemsets'], str):
        itemsets_df["itemsets"] = itemsets_df["itemsets"].apply(ast.literal_eval)

    return itemsets_df

# 推荐函数：仅返回词语（无支持度、无滑块）
def recommend_keyword(keyword, itemsets_df, top_n=5):
    keyword = keyword.strip()
    if not keyword:
        return "请输入关键词哦~"

    # 分词
    words = list(jieba.cut(keyword))
    # print(words)
    if not words:
        return "无法识别有效词语。"

    # 每个词的推荐集合
    word_to_related = {}
    for w in words:
        related = set()
        for _, row in itemsets_df.iterrows():
            items = row["itemsets"]
            if any(w in item for item in items):
                for item in items:
                    if item != w:
                        related.add(item)
        if related:
            word_to_related[w] = related

    if not word_to_related:
        return "😢 未找到任何相关词，可能是新词或词义罕见。"

    # 计算所有词的推荐交集
    sets = list(word_to_related.values())
    common_related = set.intersection(*sets) if len(sets) > 1 else sets[0]

    if not common_related:
        result = "⚠️ 各关键词无共同推荐，以下是单独推荐：\n\n"
    else:
        result = "推荐：\n" + "、".join(sorted(list(common_related)[:top_n])) + "\n\n"
        result += "分词推荐如下：\n"

    # 每个词单独展示推荐
    for word, related in word_to_related.items():
        if related:
            items = sorted(list(related))[:top_n]
            result += f"「{word}」：{'、'.join(items)}\n"

    return result

if __name__ == "__main__":
    style = "shi"
    keyword = "春雨山水话"
    df = load_data(style)
    result = recommend_keyword(keyword, df)
    print(f"推荐词语: {result}")