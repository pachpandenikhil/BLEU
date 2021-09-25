import sys
import math
import os
import typing
from collections import Counter
from os import listdir
from os.path import isfile
from typing import List

CONST_OUTPUT_FILE = "bleu_out.txt"


def read_file(file_path):
    lines = []
    with open(file_path, "r") as f:
        for line in iter(f):
            line = line.rstrip()
            if line:
                lines.append(line)
    return lines


def get_files(directory_path):
    files = []
    rel_files_path = [file for file in listdir(directory_path)]
    for file in rel_files_path:
        files.append(os.path.join(directory_path, file))
    return files


# Writes the output to file
def write_output(output, file_path):
    with open(file_path, "w") as f:
        f.write(str(output))


def ngrams(sentences, n):
    ngrams_sentences = []
    for sentence in sentences:
        ngrams_sentence = []
        sentence = sentence.split()
        for i in range(len(sentence) - n + 1):
            ngrams_sentence.append(" ".join(sentence[i : i + n]))
        ngrams_sentences.append(ngrams_sentence)
    return ngrams_sentences


def get_tokens_freq(tokens: List[str]) -> typing.Counter[str]:
    return Counter(tokens)


def compute_clip_count(
    candidate_tokens: List[List[str]], reference_tokens: List[List[List[str]]]
) -> int:
    """
    :param candidate_tokens: inner dim denotes a sentence within the candidate
    :param reference_tokens: 1st inner dim denotes a reference,
                             2nd inner dim denotes a sentence within the reference
    :return: clip count
    """
    clip_count = 0
    tot_sentences = len(candidate_tokens)
    for sent_idx in range(tot_sentences):
        cand_sent_tokens = candidate_tokens[sent_idx]
        cand_tokens_freq = get_tokens_freq(cand_sent_tokens)
        ref_tokens_freq_list = []
        for ref_tokens_list in reference_tokens:
            ref_sent_tokens = ref_tokens_list[sent_idx]
            ref_tokens_freq_list.append(get_tokens_freq(ref_sent_tokens))

        cand_tokens_clip_count = {}
        for token, token_freq in cand_tokens_freq.items():
            ref_token_freq = 0
            for ref_tokens_freq in ref_tokens_freq_list:
                if token in ref_tokens_freq:
                    ref_token_freq = max(ref_token_freq, ref_tokens_freq[token])
            cand_tokens_clip_count[token] = min(token_freq, ref_token_freq)
        clip_count += sum(cand_tokens_clip_count.values())
    return clip_count


def get_ngram_length(candidate_sentences: List[str], ngram: int) -> int:
    return sum([len(sentence.strip().split()) - ngram + 1 for sentence in candidate_sentences])


def compute_brevity_penalty(candidate_sentences, reference_sentences):
    bp = 0.0

    # getting unigram representation of candidate and reference sentences
    cand_sent_unigrams = ngrams(candidate_sentences, 1)
    ref_sent_unigrams = []
    for ref_sent_list in reference_sentences:
        ref_sent_unigrams.append(ngrams(ref_sent_list, 1))

    # computing numerator
    num = 0
    for sent_idx in range(len(candidate_sentences)):
        cand_sent_length = len(cand_sent_unigrams[sent_idx])
        closest_match = 0
        diff = float("+inf")
        for ref_sent_unigrams_list in ref_sent_unigrams:
            ref_sent_length = len(ref_sent_unigrams_list[sent_idx])
            abs_diff = abs(cand_sent_length - ref_sent_length)
            if abs_diff < diff:
                diff = abs_diff
                closest_match = ref_sent_length
        num += closest_match

    # computing denominator
    den = sum([len(cand_sent) for cand_sent in cand_sent_unigrams])

    if den > num:
        bp = 1
    else:
        exp = 1 - (float(num) / den)
        bp = math.exp(exp)

    return bp


def compute_bleu_score(candidate_sentences, reference_sentences, max_ngrams):
    bleu_score = 0.0

    # computing precision for each ngram
    ngram_precisions = []
    for ngram in range(1, max_ngrams + 1):

        # getting tokens for ngram
        candidate_tokens = ngrams(candidate_sentences, ngram)
        reference_tokens = []
        for ref_sent_list in reference_sentences:
            reference_tokens.append(ngrams(ref_sent_list, ngram))

        # computing clip count for candidate tokens
        clip_count = compute_clip_count(candidate_tokens, reference_tokens)
        cand_ngram_len = get_ngram_length(candidate_sentences, ngram)

        precision = float(clip_count) / cand_ngram_len
        ngram_precisions.append(precision)

    # computing brevity penalty
    bp = compute_brevity_penalty(candidate_sentences, reference_sentences)

    # computing BLEU score
    precision_prod = 1
    for precision in ngram_precisions:
        precision_prod *= precision

    bleu_score = math.pow(precision_prod, 0.25) * bp

    return bleu_score


def main(candidate_file="", reference_folder="", max_ngrams=4):
    bleu_score = 0.0
    if len(sys.argv) > 1 and not candidate_file and not reference_folder:
        candidate_file = sys.argv[1]
        reference_folder = sys.argv[2]

    # reading candidate file
    candidate_sentences = read_file(candidate_file)

    # reading reference file/folder
    reference_sentences = []
    if isfile(reference_folder):
        reference_sentences.append(read_file(reference_folder))
    else:
        files = get_files(reference_folder)
        for reference_file in files:
            reference_sentences.append(read_file(reference_file))

    # computing BLEU score
    bleu_score = compute_bleu_score(
        candidate_sentences, reference_sentences, max_ngrams
    )
    return bleu_score


# main execution
if __name__ == "__main__":
    bleu_score = main()
    output_file = CONST_OUTPUT_FILE
    write_output(bleu_score, output_file)
