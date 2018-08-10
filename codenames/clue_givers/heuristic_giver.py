# from codenames.clue_givers.giver import Giver
import logging
import operator
from itertools import combinations, chain
from typing import List

import numpy as np

from codenames.clue_givers.giver import Giver
from codenames.embedding_handler import EmbeddingHandler
from codenames.utils.game_utils import Clue, DEFAULT_NUM_CLUES, UNREVEALED, GOOD, BAD, CIVILIAN, ASSASSIN, DEFAULT_NUM_TARGETS, CIVILIAN_PENALTY, ASSASSIN_PENALTY, MULTIGROUP_PENALTY


class HeuristicGiver(Giver):

    def __init__(self,
                 embedding_handler: EmbeddingHandler):
        self.embedding_handler = embedding_handler

    ''' Returns list of n=NUM_CLUES of Clues for a given group of words'''
    def _get_clues(self, pos_words_subset, neg_words, civ_words, ass_words, aggressive, num_clues=DEFAULT_NUM_CLUES,MULTIGROUP_PENALTY=MULTIGROUP_PENALTY):
        clues = []
        count = len(pos_words_subset) - 1

        pos_words_vectors = self.embedding_handler.embed_words_list(pos_words_subset)
        neg_words_vectors = self.embedding_handler.embed_words_list(neg_words)
        civ_words_vectors = self.embedding_handler.embed_words_list(civ_words)
        ass_words_vectors = self.embedding_handler.embed_words_list(ass_words)

        if pos_words_vectors is None or neg_words_vectors is None:
            return None

        mean_vector = pos_words_vectors.mean(axis=0)
        mean_vector /= np.sqrt(mean_vector.dot(mean_vector))

        dotproducts = np.dot(self.embedding_handler.embedding_weights, mean_vector).reshape(-1)
        closest = np.argsort(dotproducts)[::-1]

        '''Skew 'good clues' towards larger groups of target words'''
        if aggressive:
            if count <= 1:
                MULTIGROUP_PENALTY += .1
            elif count <= 3:
                MULTIGROUP_PENALTY += .4
            else:
                MULTIGROUP_PENALTY += .7

        for i in range(num_clues):
            clue_index = closest[i]
            clue_word = self.embedding_handler.index_to_word[clue_index]

            clue_vector = self.embedding_handler.get_embedding_by_index(clue_index)

            clue_pos_words_similarities = np.dot(pos_words_vectors, clue_vector)
            clue_neg_words_similarities= np.dot(neg_words_vectors, clue_vector)
            min_clue_cosine = np.min(clue_pos_words_similarities) + MULTIGROUP_PENALTY

            #logging.info('potential clue : {}'.format(clue_word))

            max_neg_cosine = np.max(clue_neg_words_similarities)

            if max_neg_cosine >= min_clue_cosine:
                continue
            if civ_words_vectors is not None:
                clue_civ_words_similarities = np.dot(civ_words_vectors, clue_vector)
                max_civ_cosine = np.max(clue_civ_words_similarities)
                if max_civ_cosine >= min_clue_cosine - CIVILIAN_PENALTY:
                    continue
            if ass_words_vectors is not None:
                max_ass_cosine = np.dot(ass_words_vectors,clue_vector)
                if max_ass_cosine >= min_clue_cosine - ASSASSIN_PENALTY:
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
        neg_words = [board[idx] for idx, val in enumerate(allIDs) if val == BAD]
        civ_words = [board[idx] for idx, val in enumerate(allIDs) if val == CIVILIAN]
        ass_words = [board[idx] for idx, val in enumerate(allIDs) if val == ASSASSIN]

        available_targets = [word for word in pos_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_neg = [word for word in neg_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_civ = [word for word in civ_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_ass = [word for word in ass_words if
                             game_state[board.index(word)] == UNREVEALED]

        num_revealed = 0
        for idx, value in enumerate(game_state):
            if value == -1:
                num_revealed += 1
        if num_revealed > len(game_state) / 2 and score < num_revealed:
            aggressive = True
        else:
            aggressive = False

        if len(available_targets) > DEFAULT_NUM_TARGETS:
            num_words = DEFAULT_NUM_TARGETS
        else:
            num_words = len(available_targets)

        clues_by_group = []
        for count in range(num_words, 0, -1):
            for group in combinations(range(num_words), count):
                target_group = [available_targets[i] for i in group]
                clues_for_group = self._get_clues(target_group, available_neg, available_civ, available_ass, aggressive)
                if clues_for_group is not None:
                    clues_by_group.append(self._get_clues(target_group, available_neg, available_civ, available_ass, aggressive))
        clues_by_group = list(chain.from_iterable(clues_by_group))
        clues_by_group.sort(key=operator.itemgetter(1))
        clues_by_group = [clue[0] for clue in clues_by_group]
        return clues_by_group
