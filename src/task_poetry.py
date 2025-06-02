import os
import sys
import pickle
import argparse
from typing import Dict, List, Union

from data_loader import PlainDataLoader
from shingling import Shingles
from get_similarity import Get_Similarity

def load_model(dataset: str):
    """
    Load the similarity model for the specified dataset.
    """
    if dataset == 'shi':
        dataset = 'yudingquantangshi'
    if dataset not in ['songci', 'yuanqu', 'yudingquantangshi', 'chuci']:
        raise TypeError("Unsupported dataset. Supported datasets are: songci, yuanqu, yudingquantangshi, chuci.")
    
    loader = PlainDataLoader()
    poems = loader.extract_full_poem(dataset)
    
    with open(f"./models/{dataset}_model.pkl", "rb") as file:
        similarer = pickle.load(file)
    
    return poems, similarer

def recommend_poetry(text: Union[str,List[str]], poems, similarer:Get_Similarity , num:int = 3) -> List[List[str]]:
    recommended = []
    shingler = Shingles()
    text = text.split()
    text = shingler.shingling_sentence(text)
    
    for id in similarer.find_similar_poem(text,num):
        content = ''.join(poems[id])
        if len(content) > 100:
            content = content[:100] + '……'
        recommended.append(content)

    return '\n\n'.join(recommended)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="山水风光", help='words or sentences already written')
    parser.add_argument('--num', type=int, default=3, help='number of poems you want to recommend for you')

    parser.add_argument('--dataset', type=str, default="songci" ,help='style you want')
    args = parser.parse_args()
    peoms, similarer = load_model(args.dataset)
    recommended = recommend_poetry(args.text, peoms, similarer, args.num)
    print(recommended)