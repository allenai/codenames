from gensim.models import KeyedVectors
from itertools import combinations, chain
from codenames.embedding_handler import EmbeddingHandler
from codenames.clue_givers.giver import Giver
import numpy as np
from codenames.utils.game_utils import Clue
import operator
from typing import List
import logging
from sklearn.metrics.pairwise import cosine_similarity

class HeuristicGiver2(Giver):

    def __init__(self, board: [str],
                 allIDs: List[int],
                 embeddinghandler: EmbeddingHandler,
                 NUM_CLUES: int=50):
        super().__init__(board, allIDs)
        self.assassin = [board[idx] for idx, val in enumerate(allIDs) if val == 0]
        self.pos_words = [board[idx] for idx, val in enumerate(allIDs) if val == 1]
        self.neg_words = [board[idx] for idx, val in enumerate(allIDs) if val == 2]
        self.civilians = [board[idx] for idx, val in enumerate(allIDs) if val == 3]
        self.embedding_handler = embeddinghandler
        self.NUM_CLUES = NUM_CLUES

    ''' Returns list of n=NUM_CLUES of Clues for a given group of words'''
    def _get_clues(self, group, assassin, neg_words, civilians, aggressive, NUM_CLUES):
        clues = []
        count = len(group)

        clue_indices = [self.embedding_handler.vocab[word].index for word in group]
        clue_vectors = self.embedding_handler.syn0[clue_indices]
        neg_indices = [self.embedding_handler.vocab[word].index for word in neg_words]
        neg_vectors = self.embedding_handler.syn0[neg_indices]
        civilian_indices = [self.embedding_handler.vocab[word].index for word in civilians]
        civilian_vectors = self.embedding_handler.syn0[civilian_indices]
        assassin_indices = [self.embedding_handler.vocab[word].index for word in assassin]
        assassin_vectors = self.embedding_handler.syn0[assassin_indices]
        
        mean_vector = clue_vectors.mean(axis=0)
        mean_vector /= np.sqrt(mean_vector.dot(mean_vector))

        cosines = cosine_similarity(self.embedding_handler.syn0, mean_vector.reshape(1,-1)).flatten()
        closest = np.argsort(cosines)[::-1]
        civilian_penalty = .05
        assassin_penalty = .08

        '''Skew 'good clues' towards larger groups of target words'''
        if aggressive and count > 1 and count < 3:
            multigroup_penalty = .4
        elif aggressive and count > 2:
            multigroup_penalty = .8
        else:
            multigroup_penalty = 0

        for i in range(NUM_CLUES):
            clue = None
            clue_index = closest[i]
            clue = self.embedding_handler.index2word[clue_index]
            if clue.lower() in group:
                continue
            clue_vector = self.embedding_handler.syn0[clue_index].reshape(1,-1)
            clue_cosine = cosine_similarity(clue_vectors, clue_vector).flatten()
            min_clue_cosine = np.min(clue_cosine) + multigroup_penalty


            if neg_words:
                neg_cosine = cosine_similarity(neg_vectors, clue_vector)
                max_neg_cosine = np.max(neg_cosine)
                if max_neg_cosine >= min_clue_cosine:
                    continue

            if civilians:
                civilian_cosine = cosine_similarity(civilian_vectors, clue_vector).flatten()
                max_civ_cosine = np.max(civilian_cosine)
                if max_civ_cosine >= min_clue_cosine - civilian_penalty:
                    continue
            if assassin:
                assassin_cosine = cosine_similarity(assassin_vectors, clue_vector).flatten()
                max_ass_cosine = np.max(assassin_cosine)
                if max_ass_cosine >= min_clue_cosine- assassin_penalty:
                    continue
            clues.append((Clue(clue.lower(), group, count),min_clue_cosine))
        return clues
    def _unique_clues(self, ordered_list_of_clues):
        clue_words = []
        list_of_clues = []
        for clue in ordered_list_of_clues:
            if clue[0].clue_word not in clue_words:
                clue_words.append(clue[0].clue_word)
                list_of_clues.append(clue)
        return list_of_clues
    '''List of Clues sorted by descending Cosine distance'''
    def get_next_clue(self, game_state: List[int],
                      score: int):
        available_targets = [word for word in self.pos_words if game_state[self.board.tolist().index(word)] == -1]
        available_civilians = [word for word in self.civilians if game_state[self.board.tolist().index(word)] == -1]
        available_neg_words = [word for word in self.neg_words if game_state[self.board.tolist().index(word)] == -1]
        available_assassins = [word for word in self.assassin if game_state[self.board.tolist().index(word)] == -1]
        num_revealed = 0
        for idx, value in enumerate(game_state):
            if value == -1:
                num_revealed += 1
        if num_revealed > len(game_state)/2 and score < num_revealed:
            aggressive = True
        else:
            aggressive = False
        all_clues = []
        num_words = len(available_targets)
        for count in range(num_words, 0, -1):
            for group in combinations(range(num_words),count):
                logging.info(group, self.neg_words)
                target_group = [available_targets[i] for i in group]
                all_clues.append(self._get_clues(available_targets, available_assassins, available_neg_words, available_civilians, aggressive, self.NUM_CLUES))
        all_clues = list(chain.from_iterable(all_clues))
        all_clues.sort(key=operator.itemgetter(1),reverse=True)
        all_clues = self._unique_clues(all_clues)
        all_clues = [clue[0] for clue in all_clues]
        return all_clues


def main():
    test_embed = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300-SLIM.bin',binary=True)
    test_board = ["water", "notebook", "board", "boy", "shoe", "cat", "pear", "sandwich","chair","pants","phone","internet"]
    test_allIDs = [1, 2, 2, 1, 3, 1, 2, 3,1,1,0,1]
    cg = HeuristicGiver(test_board, test_allIDs, test_embed)


if __name__ == "__main__":
    main()
