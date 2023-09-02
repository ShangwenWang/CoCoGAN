# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU

# -*- coding:utf-8 -*-
import argparse
import os
from typing import List

from evaluator.CodeBLEU import (bleu, dataflow_match, syntax_match,
                                weighted_ngram_match)

def evaluate_per_example(
        reference, hypothesis, lang, params='0.25,0.25,0.25,0.25'
):
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]
    hypothesis = [hypothesis]
    pre_references = [[reference]]
    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])
    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)
    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]
    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
    # calculate weighted ngram match
    root_dir = os.path.dirname(__file__)
    keywords = [x.strip() for x in open(root_dir + '/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)
    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)
    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)
    # dataflow_match_score = dataflow_match.my_dataflow_match(references, hypothesis, lang)
    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.
          format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
    codebleu = alpha * ngram_match_score \
               + beta * weighted_ngram_match_score \
               + gamma * syntax_match_score \
               + theta * dataflow_match_score
    return {
        'em': 1. if reference.strip() == hypothesis[0].strip() else 0.,
        'bleu': ngram_match_score,
        'wbleu': weighted_ngram_match_score,
        'syntax': syntax_match_score,
        'dataflow': dataflow_match_score,
        'codebleu': codebleu
    }


def get_codebleu(refs:List[str], hyp:List[str], lang, params='0.25,0.25,0.25,0.25'):
    if not isinstance(refs, list):
        refs = [refs]
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]
    root_dir = os.path.dirname(__file__)

    # preprocess inputs
    ref_exist_flag = True
    for ref in refs:
        if not os.path.exists(ref):
            ref_exist_flag = False
    if ref_exist_flag is True:
        pre_references = [[x.strip() for x in open(file, 'r', encoding='utf-8').readlines()] for file in refs]
    elif len(refs) > 0 and not isinstance(refs[0], list):
        pre_references = [[ref.strip() for ref in refs]]
    else:
        pre_references = refs
    if isinstance(hyp, str) and os.path.exists(hyp):
        hypothesis = [x.strip() for x in open(hyp, 'r', encoding='utf-8').readlines()]
    else:
        hypothesis = [hy.strip() for hy in hyp]
    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    
    keywords = [x.strip() for x in open(root_dir + '/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)
    # dataflow_match_score = dataflow_match.my_dataflow_match(references, hypothesis, lang)

    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.
          format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    codebleu = alpha * ngram_match_score \
               + beta * weighted_ngram_match_score \
               + gamma * syntax_match_score \
               + theta * dataflow_match_score

    return codebleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', type=str, nargs='+', required=True,
                        help='reference files')
    parser.add_argument('--hyp', type=str, required=True,
                        help='hypothesis file')
    parser.add_argument('--lang', type=str, required=True,
                        choices=['java', 'js', 'c_sharp', 'php', 'go', 'python', 'ruby'],
                        help='programming language')
    parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                        help='alpha, beta and gamma')

    args = parser.parse_args()
    code_bleu_score = get_codebleu(args.refs, args.hyp, args.lang, args.params)
    print('CodeBLEU score: ', code_bleu_score)
