import os
import sys
import random
import numpy as np
from typing import Dict, List
from collections import defaultdict

from data_loader import PlainDataLoader
from shingling import Shingles



MAX_SHINGLEID = 2**32 - 1
NEXTPRIME = 4294967311
class Get_Similarity:
    def __init__(self, shingles:Dict, num_hash:int=10):
        self.num_poems = len(shingles)
        self.shingles = shingles
        self.num_hash = num_hash
        self.a_list = random.sample(range(1,MAX_SHINGLEID), num_hash)
        self.b_list = random.sample(range(1,MAX_SHINGLEID), num_hash)
        self.signatures = defaultdict(List)
        for id, shingles in self.shingles.items():
            signature = []
            for i in range(self.num_hash):
                a = self.a_list[i]
                b = self.b_list[i]
                minhash = float('inf')
                for shingle in shingles:
                    value = (a*shingle+b)%NEXTPRIME 
                    minhash = min(value,minhash)
                signature.append(minhash)
            self.signatures[id] = signature
        
    def find_most_similar_of_poem(self, id:int, n:int) -> List[int]:
        sim_of_poems = defaultdict(int)
        signature1 = self.signatures[id]
        for i in range(self.num_poems):
            if i == id:
                continue
            signature2 = self.signatures[i]
            num_match = sum(1 for x, y in zip(signature1,signature2) if x==y)
            sim_of_poems[i] = num_match
        sorted_sims = sorted(sim_of_poems.items(), key=lambda x: x[1], reverse=True)
        return [d[0] for d in sorted_sims[:n]]
    
    def find_similar_poem(self, shingles:set, n:int) -> List[int]:
        signature = []
        for i in range(self.num_hash):
            a = self.a_list[i]
            b = self.b_list[i]
            minhash = float('inf')
            for shingle in shingles:
                value = (a*shingle+b)%NEXTPRIME 
                minhash = min(value,minhash)
            signature.append(minhash)
        sim_of_poems = defaultdict(int)
        for id, sig in self.signatures.items():
            num_match = sum(1 for x, y in zip(signature,sig) if x==y)
            sim_of_poems[id] = num_match
        sorted_sims = sorted(sim_of_poems.items(), key=lambda x: x[1], reverse=True)
        return [d[0]for d in sorted_sims[:n]]
    

if __name__ == "__main__":
    loader = PlainDataLoader()
    poems = loader.extract_full_poem("songci")
    shingler = Shingles()
    shingler.shingling(poems)
    shingles = shingler.get_item()
    similarer = Get_Similarity(shingles, 100)
    sim_id = similarer.find_most_similar_of_poem(1, 5)
    print(poems[1])
    for id in sim_id:
        print(poems[id])
    # test = shingler.shingling_sentence(['归去', '烟雨', '春风'])
    # for id in similarer.find_similar_poem(test,5):
    #     print(poems[id])
