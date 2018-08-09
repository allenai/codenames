from overrides import overrides
#from codenames.clue_givers.giver import Giver
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from itertools import combinations, chain
from collections import namedtuple
from codenames.embedding_handler import EmbeddingHandler
from codenames.clue_givers.giver import Giver
import numpy as np
import sklearn.cluster
from codenames.utils.game_utils import get_available_choices, Clue
import operator
from typing import List
import logging
#import EmbeddingHandler as embed
#model = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300-SLIM.bin',binary=True)
#Clue = namedtuple('Clue', ['clue_word', 'intended_board_words', 'count'])

class ClueGiver(Giver):

    def __init__(self, board: [str],
                 allIDs: List[int],
                 embeddinghandler: EmbeddingHandler):
        #self.board = board
        super().__init__(board, allIDs)
        self.pos_words = [board[idx] for idx, val in enumerate(allIDs) if val == 1]
        #TODO consider neutral words and assassin
        self.neg_words = [word for word in board if word not in self.pos_words]
        #self.model = embeddinghandler.embedding
        self.embedding_handler = embeddinghandler

    NUM_CLUES = 10
    ''' Returns list of n=NUM_CLUES of Clues for a given group of words'''
    def _get_clues(self, group, neg_words, num_clues=NUM_CLUES):
        clues = []
        count = len(group)
        
        clue_indices = [self.embedding_handler.vocab[word].index for word in group]
        clue_vectors = self.embedding_handler.syn0[clue_indices]
        neg_indices = [self.embedding_handler.vocab[word].index for word in neg_words]
        neg_vectors = self.embedding_handler.syn0[neg_indices]
        
        mean_vector = clue_vectors.mean(axis=0)
        mean_vector /= np.sqrt(mean_vector.dot(mean_vector))

        cosines = np.dot(self.embedding_handler.syn0[:, np.newaxis], mean_vector).reshape(-1) #shape (vocab_size,)
        closest = np.argsort(cosines)[::-1]
        

        for i in range(num_clues):
            clue = None
            clue_index = closest[i]
            clue = self.embedding_handler.index2word[clue_index]
            clue_vector = self.embedding_handler.syn0[clue_index]
            clue_cosine = np.dot(clue_vectors[:, np.newaxis], clue_vector)
            neg_cosine = np.dot(neg_vectors[:, np.newaxis], clue_vector)
            min_clue_cosine = np.min(clue_cosine)
            logging.info('potential clue')
            logging.info(clue)
            max_neg_cosine = np.max(neg_cosine)
            if max_neg_cosine >= min_clue_cosine:
                continue
            clues.append((Clue(clue, group, count),np.mean(clue_cosine)))
        return clues

    '''List of Clues sorted by descending Cosine distance'''
    def get_next_clue(self, game_state: List[int],
                      score: int):
        logging.info(self.pos_words)
        logging.info(self.neg_words)
        available_targets = [word for word in self.pos_words if game_state[self.board.index(word)] != -1]
        logging.info(available_targets)
        all_clues = []
        num_words = len(available_targets)
        for count in range(num_words, 0, -1):
            for group in combinations(range(num_words),count):
                logging.info(group, self.neg_words)
                target_group = [available_targets[i] for i in group]
                logging.info('target words')
                logging.info(target_group)
                all_clues.append(self._get_clues(target_group, self.neg_words))
        all_clues = list(chain.from_iterable(all_clues))
        all_clues.sort(key=operator.itemgetter(1))
        all_clues = [clue[0] for clue in all_clues]
        return all_clues


def main():
    #test_embed = EmbeddingHandler("./test_embeds.p")
    test_embed = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300-SLIM.bin',binary=True)
    #import pdb; pdb.set_trace()
    test_board = ["woman", "man", "girl", "boy", "blue", "cat", "queen", "king"]
    test_allIDs = [1, 2, 2, 1, -1, 1, 2, 3]
    test_target = ["woman", "boy"]
    cg = ClueGiver(test_board, test_allIDs, test_embed)
    logging.info('cllllue')
    logging.info(cg.get_next_clue([-1, 2, 2, 1, -1, 1, 2, 3], 3))


if __name__ == "__main__":
    main()
