from unittest import TestCase

from codenames.guessers.learned_guesser_game_state import LearnedGuesserGameState
from codenames.embedding_handler import EmbeddingHandler
from codenames.guessers.policy.similarity_threshold_game_state import SimilarityThresholdGameStatePolicy


class TestLearnedGuesser(TestCase):
    def test_guess(self):
        embedding_handler = EmbeddingHandler("tests/fixtures/sample_embedding.txt")
        sample_board = ["boy", "girl", "woman", "man"]
        embed_size = list(embedding_handler.embedding.values())[0].shape[0]
        policy = SimilarityThresholdGameStatePolicy(embed_size)
        guesser = LearnedGuesserGameState(sample_board, embedding_handler, policy, 0.1)
        sample_state = [0, -1, -1, -1]
        guesses = guesser.guess("boy", 1, sample_state, 0)
        guesser.report_reward([10])
        print(guesses)
