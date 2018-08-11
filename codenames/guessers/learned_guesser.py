from typing import List
from overrides import overrides

import torch
from torch.distributions import Categorical

from codenames.guessers.guesser import Guesser
from codenames.guessers.policy.guesser_policy import GuesserPolicy
from codenames.embedding_handler import EmbeddingHandler
from codenames.utils import game_utils as util
from codenames.guessers.policy.similarity_threshold_game_state import SimilarityThresholdGameStatePolicy

class LearnedGuesser(Guesser):
    def __init__(self,
                 embedding_handler: EmbeddingHandler,
                 policy: GuesserPolicy,
                 learning_rate: float,
                 train: bool=False) -> None:
        self.policy = policy
        self.guess_history = None
        self.guess_log_probs = None
        self.embedding_handler = embedding_handler
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.train = train

    @overrides
    def guess(self,
              board: List[str],
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
        for option in util.get_available_choices(board, game_state):
            option_vector = self.embedding_handler.get_word_vector(option)
            if option_vector is not None:
                option_vectors.append(option_vector)
                known_options.append(option)
        if not option_vectors:
            return []


        # Checks type of policy to see if we should parameterize
        if type(self.policy) == SimilarityThresholdGameStatePolicy:
            # Parameterizes game state for policy
            parameterized_game_state = []
            for i in range(4):
                parameterized_game_state.append(game_state.count(i))
            # Sample from policy
            policy_output = self.policy(torch.Tensor(clue_vector),
                                        torch.Tensor(option_vectors),
                                        torch.Tensor(parameterized_game_state))
        else:
            # Sample from policy
            policy_output = self.policy(torch.Tensor(clue_vector),
                                        torch.Tensor(option_vectors))

        distribution = Categorical(policy_output)
        predictions = distribution.sample(torch.Size((count+1,)))

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

    '''
    Save is a path of where to save if we want to save the model
    '''
    def report_reward(self,
                      rewards: List[int],
                      save: str=None) -> None:
        if self.train:
            if self.guess_log_probs is None:
                raise RuntimeError("Haven't made any guesses yet!")
            loss = torch.mul(torch.sum(torch.mul(torch.Tensor(rewards),
                                                 torch.stack(self.guess_log_probs))), -1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if save:
                torch.save(self.policy, save)
