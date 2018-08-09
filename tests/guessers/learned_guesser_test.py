from unittest import TestCase

from codenames.guessers.learned_guesser import LearnedGuesser
from codenames.embedding_handler import EmbeddingHandler
from codenames.guessers.policy.similarity_threshold import SimilarityThresholdPolicy


class TestLearnedGuesser(TestCase):
    def test_guess(self):
        embedding_handler = EmbeddingHandler("tests/fixtures/sample_embedding.pkl")
        sample_board = ["boy", "girl", "woman", "man"]
        embed_size = list(embedding_handler.embedding.values())[0].shape[0]
        policy = SimilarityThresholdPolicy(embed_size)
        guesser = LearnedGuesser(sample_board, embedding_handler, policy, 0.1)
        sample_state = [0, -1, -1, -1]
        guesses = guesser.guess("boy", 1, sample_state, 0)
        assert guesses == ['girl', 'woman']  # count + 1
