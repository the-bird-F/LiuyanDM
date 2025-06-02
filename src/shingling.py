import os
import sys
import re
from typing import List, Union, Set
from collections import defaultdict
from hashlib import sha1


from data_loader import PlainDataLoader

class Shingles:
    def __init__(self):
        self.shingles = defaultdict(set)
        self.total_shingles = set()
    
    def shingling(self, poems: List[List[str]], k:int = 2):
        for idx,poem in enumerate(poems):
            for sentence in poem:
                words = re.split(r'[，。！？,.;!?、]', sentence)
                for word in words:
                    if len(word) <= k and '□' not in word:
                        self.total_shingles.add(word)
                        hashvalue = sha1(word.encode())
                        hashvalue = int.from_bytes(hashvalue.digest()[:4] , byteorder='big')
                        self.shingles[idx].add(hashvalue)
                    else:
                        tmp = []
                        for i in range(k):
                            tmp.append(word[0+i:len(word)-k+i+1])
                        for shingle in zip(*tmp):
                            if '□' in shingle:
                                continue
                            shingle = ''.join(shingle)
                            self.total_shingles.add(shingle)
                            hashvalue = sha1(shingle.encode())
                            hashvalue = int.from_bytes(hashvalue.digest()[:4] , byteorder='big')
                            self.shingles[idx].add(hashvalue)

    def get_item(self):
        return self.shingles
    
    def get_num_shingles(self):
        return len(self.total_shingles)
    
    def shingling_sentence(self, sentence: Union[str, List[str]], k:int=2) -> Set[int]:
        shingles = set()
        if isinstance(sentence, str):
            words = re.split(r'[，。！？,.;!?、]', sentence)
            for word in words:
                if len(word) <= k and '□' not in word:
                    hashvalue = sha1(word.encode())
                    hashvalue = int.from_bytes(hashvalue.digest()[:4] , byteorder='big')
                    shingles.add(hashvalue)
                else:
                    tmp = []
                    for i in range(k):
                        tmp.append(word[0+i:len(word)-k+i+1])
                    for shingle in zip(*tmp):
                        if '□' in shingle:
                            continue
                        shingle = ''.join(shingle)
                        hashvalue = sha1(shingle.encode())
                        hashvalue = int.from_bytes(hashvalue.digest()[:4] , byteorder='big')
                        shingles.add(hashvalue)
        elif isinstance(sentence, List):
            for s in sentence:
                words = re.split(r'[，。！？,.;!?、]', s)
                for word in words:
                    if len(word) <= k and '□' not in word:
                        hashvalue = sha1(word.encode())
                        hashvalue = int.from_bytes(hashvalue.digest()[:4] , byteorder='big')
                        shingles.add(hashvalue)
                    else:
                        tmp = []
                        for i in range(k):
                            tmp.append(word[0+i:len(word)-k+i+1])
                        for shingle in zip(*tmp):
                            if '□' in shingle:
                                continue
                            shingle = ''.join(shingle)
                            hashvalue = sha1(shingle.encode())
                            hashvalue = int.from_bytes(hashvalue.digest()[:4] , byteorder='big')
                            shingles.add(hashvalue)

        return shingles
    
if __name__ == '__main__':
    loader = PlainDataLoader()
    poems = loader.extract_full_poem("songci")
    shingler = Shingles()
    shingler.shingling(poems)
    print(len(poems))
    print(len(shingler.shingles))
    print(shingler.shingling_sentence('迟迟'))
