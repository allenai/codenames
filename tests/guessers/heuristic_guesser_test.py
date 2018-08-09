from unittest import TestCase

from codenames.guessers.heuristic_guesser import HeuristicGuesser
from codenames.embedding_handler import EmbeddingHandler


class TestHeuristicGuesser(TestCase):
    def test_guess(self):
        embedding_handler = EmbeddingHandler("tests/fixtures/sample_embedding.pkl")
        sample_board = ["boy", "girl", "woman", "man"]
        guesser = HeuristicGuesser(sample_board, embedding_handler)
        sample_state = [0, -1, -1, -1]
        guesses = guesser.guess("boy", 1, sample_state, 0)
        assert guesses == ['man', 'woman']  # count + 1
