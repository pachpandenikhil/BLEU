import unittest

from calculatebleu import compute_clip_count, main, get_tokens_freq, get_ngram_length


class TestCalculateBleu(unittest.TestCase):
    def test_compute_clip_count(self):
        candidate_tokens = [["the", "the", "the", "the", "the", "the", "the"]]
        reference_tokens = [
            [["the", "cat", "is", "on", "the", "mat"]],
            [["there", "is", "a", "cat", "on", "the", "mat"]],
        ]
        assert 2 == compute_clip_count(candidate_tokens, reference_tokens)

    def test_main(self):
        assert 0.7311 == round(main("../candidate.txt", "../references", 1), 4)

    def test_get_tokens_freq(self):
        tokens = ["the", "the", "the", "the", "the", "the", "the"]
        assert {"the": 7} == get_tokens_freq(tokens)

    def test_get_ngram_length(self):
        candidate_sentences = ["the cat is on the mat", "there is a cat on the mat"]
        ngram = 1
        assert 13 == get_ngram_length(candidate_sentences, ngram)
