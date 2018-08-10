# from codenames.clue_givers.giver import Giver
import logging
import operator
from itertools import combinations, chain
from typing import List

import numpy as np

from codenames.clue_givers.giver import Giver
from codenames.embedding_handler import EmbeddingHandler
from codenames.utils.game_utils import Clue, DEFAULT_NUM_CLUES, UNREVEALED, GOOD


class HeuristicGiver(Giver):

    def __init__(self,
                 embedding_handler: EmbeddingHandler):
        self.embedding_handler = embedding_handler

    ''' Returns list of n=NUM_CLUES of Clues for a given group of words'''
    def _get_clues(self, pos_words_subset, neg_words, num_clues=DEFAULT_NUM_CLUES):
        clues = []
        count = len(pos_words_subset)

        pos_words_vectors = self.embedding_handler.embed_words_list(pos_words_subset)
        neg_words_vectors = self.embedding_handler.embed_words_list(neg_words)

        if pos_words_vectors is None or neg_words_vectors is None:
            return None

        mean_vector = pos_words_vectors.mean(axis=0)
        mean_vector /= np.sqrt(mean_vector.dot(mean_vector))

        # shape (vocab_size,)
        cosines = np.dot(self.embedding_handler.embedding_weights, mean_vector).reshape(-1)
        closest = np.argsort(cosines)[::-1]

        for i in range(num_clues):
            clue_index = closest[i]
            clue_word = self.embedding_handler.index_to_word[clue_index]

            clue_vector = self.embedding_handler.get_embedding_by_index(clue_index)

            clue_pos_words_similarities = np.dot(pos_words_vectors, clue_vector)
            clue_neg_words_similarities= np.dot(neg_words_vectors, clue_vector)
            min_clue_cosine = np.min(clue_pos_words_similarities)

            #logging.info('potential clue : {}'.format(clue_word))

            max_neg_cosine = np.max(clue_neg_words_similarities)

            if max_neg_cosine >= min_clue_cosine:
                continue
            clues.append((Clue(clue_word, pos_words_subset, count),
                          np.mean(clue_pos_words_similarities)))
        return clues

    '''List of Clues sorted by descending Cosine distance'''

    def get_next_clue(self,
                      board: List[str],
                      allIDs: List[int],
                      game_state: List[int],
                      score: int):
        pos_words = [board[idx] for idx, val in enumerate(allIDs) if val == GOOD]
        # TODO consider neutral words and assassin
        neg_words = [word for word in board if word not in pos_words]
        available_targets = [word for word in pos_words if
                             game_state[board.index(word)] == UNREVEALED]
        clues_by_group = []
        num_words = len(available_targets)
        for count in range(num_words, 0, -1):
            for group in combinations(range(num_words), count):
                target_group = [available_targets[i] for i in group]
                clues_for_group = self._get_clues(target_group, neg_words)
                if clues_for_group is not None:
                    clues_by_group.append(self._get_clues(target_group, neg_words))
        clues_by_group = list(chain.from_iterable(clues_by_group))
        clues_by_group.sort(key=operator.itemgetter(1))
        clues_by_group = [clue[0] for clue in clues_by_group]
        return clues_by_group


def main():
    # test_embed = EmbeddingHandler("./test_embeds.p")
    test_embed = EmbeddingHandler('tests/fixtures/model.txt')
    # import pdb; pdb.set_trace()
    test_board = ["woman", "man", "girl", "boy", "blue", "cat", "queen", "king"]
    test_allIDs = [1, 2, 2, 1, -1, 1, 2, 3]
    test_target = ["woman", "boy"]
    cg = HeuristicGiver(test_board, test_allIDs, test_embed)
    logging.info('cllllue')
    logging.info(cg.get_next_clue([-1, 2, 2, 1, -1, 1, 2, 3], 3))


if __name__ == "__main__":
    main()
