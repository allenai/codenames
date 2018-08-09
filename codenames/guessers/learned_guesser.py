from typing import List
from overrides import overrides

import torch
from torch.distributions import Categorical

from codenames.guessers.guesser import Guesser
from codenames.guessers.policy.policy import Policy
from codenames.embedding_handler import EmbeddingHandler
from codenames.utils import game_utils as util


class LearnedGuesser(Guesser):
    def __init__(self,
                 board: List[str],
                 embedding_handler: EmbeddingHandler,
                 policy: Policy,
                 learning_rate: float) -> None:
        super(LearnedGuesser, self).__init__(board, embedding_handler)
        self.policy = policy
        self.guess_history = None
        self.guess_log_probs = None
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    @overrides
    def guess(self,
              clue: str,
              count: int,
              game_state: List[int],
              current_score: int) -> List[str]:
        # Get vectors for clues and options
        clue_vector = self.embedding_handler.get_word_vector(clue)
        if clue_vector is None:
            return []
        option_vectors = []
        known_options = []
        for option in util.get_available_choices(self.board, game_state):
            option_vector = self.embedding_handler.get_word_vector(option)
            if option_vector is not None:
                option_vectors.append(option_vector)
                known_options.append(option)
        if not option_vectors:
            return []

        # Sample from policy
        policy_output = self.policy(torch.Tensor(clue_vector),
                                    torch.Tensor(option_vectors))
        print(policy_output)
        distribution = Categorical(policy_output)
        predictions = distribution.sample(torch.Size((count,)))

        # Return guesses
        guesses = [known_options[int(prediction)] for prediction in predictions]
        log_probs = [distribution.log_prob(prediction) for prediction in predictions]
        # Since we sampled multiple terms, there can be repetitions. We need to return unique ones.
        # We also need to ensure the order of guesses is not changed. Note that the following logic
        # may return less than `count` number of guesses.
        unique_guesses = []
        unique_guesses_log_probs = []
        seen_guesses = set()
        for guess, log_prob in zip(guesses, log_probs):
            if not guess in seen_guesses:
                unique_guesses.append(guess)
                unique_guesses_log_probs.append(log_prob)
                seen_guesses.add(guess)
        self.guess_history = unique_guesses
        self.guess_log_probs = unique_guesses_log_probs
        return unique_guesses

    def report_reward(self,
                      rewards: List[int]) -> None:
        if self.guess_log_probs is None:
            raise RuntimeError("Haven't made any guesses yet!")
        loss = torch.mul(torch.sum(torch.mul(torch.Tensor(rewards),
                                             torch.Tensor(self.guess_log_probs))), -1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
