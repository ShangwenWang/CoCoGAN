import json
from collections import Counter
from typing import List
import os
# 1. Import CrystalBLEU
from crystalbleu import corpus_bleu
from nltk.util import ngrams
from tree_sitter import Language, Parser

from evaluator.CodeBLEU.parser import index_to_code_token, tree_to_token_index

root_dir = os.path.dirname(__file__)

def tokenize(code, parser, lang):
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    return code_tokens

    
def cal_crystalbleu(references:List[str], candidates:List[str], datapath:str, language:str):
    LANGUAGE = Language(root_dir + '/CodeBLEU/parser/languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    if len(references) > 0 and not isinstance(references[0], list):
        references = [[ref] for ref in references]
        
    tokenized_hyps = [tokenize(x, parser, language) for x in candidates]
    tokenized_refs = [[tokenize(x, parser, language) for x in reference] for reference in references]
    
    # 2. Extract trivially shared n-grams
    k = 500
    with open(datapath) as f:
        data = list(map(lambda x: json.loads(x)['code'].split(),f.read().split('\n')[:-1]))
    # <tokenized_corpus> is a list of strings
    
    all_ngrams = []
    for tokenized_corpus in data:
    # Extract all n-grams of length 1-4
        for n in range(1, 5):
            all_ngrams.extend(list(ngrams(tokenized_corpus, n)))
    # Calculate frequencies of all n-grams
    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = dict(frequencies.most_common(k))

    # 3. Calculate CrystalBLEU
    crystalBLEU_score = corpus_bleu(
        tokenized_refs, tokenized_hyps, ignoring=trivially_shared_ngrams)
    return crystalBLEU_score
