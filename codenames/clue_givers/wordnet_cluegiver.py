from overrides import overrides
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from itertools import combinations, chain, permutations
from codenames.clue_givers.giver import Giver
import numpy as np
from codenames.utils.game_utils import get_available_choices, Clue
import operator
from typing import List
import logging
from nltk.corpus import wordnet as wn

class WordnetClueGiver(Giver):

    '''
    Clue giver is initialized with the board and the IDs of cards
    '''
    def __init__(self,
                 NUM_CLUES: int=10):
        #self.board = board
        super().__init__()#board, allIDs)
        #positive words
        #self.pos_words = [board[idx] for idx, val in enumerate(allIDs) if val == 1]
        #negative words
       # self.neg_words = [word for word in board if word not in self.pos_words]
        #minimum number of clues we want our get_clues method to generate
        self.NUM_CLUES = NUM_CLUES


    ''' 
    Internal method, used to get clues for a given group of pos and neg words.
    Parameters
        group : the list of positive words still in play
        neg_words: the list of negative words still in play
        num_clues: the minimum number of clues to generate.  Default = 10
        epsilon: the weight we want put in the scoring function.  Scoring function considers
            both similarity and number of cards in a clue.  Epsilon is the weight given to the 
            number of clues, and (1-Epsilon) is the weight given to the similarity of the clue word.
            Default = 0.8
    '''

    def _get_clues(self, group, neg_words, epsilon=0.8):
        clues = []

        #num starts off trying to give a hint for all positive words then moves down
        num = len(group)

        #potential permutations of positive cards of size num
        potentials = list(permutations(group, num))

        #placeholder variable for the location we are in our list of potential groups
        place_in_potentials = 0

        done = False
        while not done:
            #grabs potential group of positive clues
            pot_clue_group = potentials[place_in_potentials]

            #synsets are the positive card synsets.  Restricted to the case where the first
            #lemma is the seed word, otherwise we get weird clues.
            synsets = []
            for word in pot_clue_group:
                curr_comm_synsets = wn.synsets(word)
                curr_comm_synsets = [t for t in curr_comm_synsets if t.lemmas()[0].name() == word]
                synsets.append(curr_comm_synsets)

            #neg_synsets are the wordnet synsets for negative cards on the board
            neg_synsets = []
            for word in neg_words:
                curr_comm_synsets = wn.synsets(word)
                curr_comm_synsets = [t for t in curr_comm_synsets if t.lemmas()[0].name() == word]
                neg_synsets.extend(curr_comm_synsets)

            #dictionary variable, used to capture the score of a potential clue
            clue_scorer = {}

            #list used to store common synsets
            common_synsets = []
            pos1 = 0
            pos2 = 0

            #we need to iterate through synsets in two stages since we can only get pairwise comparisons
            for synsetgroup1 in synsets:
                common_synsets.append([])
                pos1 = 0
                for synset1 in synsetgroup1:
                    pos1 += 1
                    for synsetgroup2 in synsets:
                        pos2 = 0
                        curr_comm_synsets = []
                        for synset2 in synsetgroup2:
                            pos2 += 1

                            #if we aren't doing single word clues, we don't want to compare the same synsets
                            if synset1 == synset2 and num > 1:
                                continue

                            #the current common synsets for these two instances of synsets
                            curr_comm = synset1.common_hypernyms(synset2)
                            for common_ancester in curr_comm:
                                is_neg = False

                                #limits out synsets which are in negative word's hierarchy
                                for neg_synset in neg_synsets:
                                    if common_ancester in neg_synset.common_hypernyms(neg_synset) or common_ancester == neg_synset:
                                        is_neg = True
                                if not is_neg:
                                    #gets similarity between clue + synset1
                                    len1 = common_ancester.path_similarity(synset1)
                                    curr_comm_synsets.append(common_ancester)

                                    #adds synset to dict if not already there
                                    if common_ancester not in clue_scorer:
                                        clue_scorer[common_ancester] = []

                                    #adds the score for the synset to the dict, according to scroing function
                                    clue_scorer[common_ancester].append((1-epsilon) * (len(synsetgroup1) - pos1) * len1 + (epsilon * num))

                                    #want to do for second synset if not the same
                                    if synset1 != synset2:
                                        len2 = common_ancester.path_similarity(synset2)
                                        clue_scorer[common_ancester].append((1-epsilon) * (len(synsetgroup2) - pos2) * len2 + (epsilon * num))
                        common_synsets[-1].extend(curr_comm_synsets)

            #now need to determine the intersection among the pairs of similar words so we can be sure that a clue
            #applies to all words
            intersect = set(common_synsets[0])
            for s in common_synsets[1:]:
                intersect.intersection_update(s)

            #if this is true, we found a clue
            if len(intersect) != 0:
                #for each clue in the set,
                for clue in intersect:

                    #Want to go through all lemmas since those are ways of expressing a synset
                    for lemma in clue.lemmas():

                        #filters out multi-words
                        if "_" not in lemma.name() and lemma.name() not in group:
                            #appends clue along with normalized score
                            clues.append((Clue(lemma.name(), pot_clue_group, num), float(np.sum(clue_scorer[clue])) / float(len(clue_scorer[clue]))))

            #advances to the next group of num words
            place_in_potentials += 1

            #if we need to decrement num and get new permutations
            if place_in_potentials >= len(potentials):
                num -= 1

                #break conditions, when we've tried all combos or reached our minimum
                if num < 1 or len(clues) >= self.NUM_CLUES:
                    break

                #if we don't break, we get new permutations and reset our place
                potentials = list(permutations(group, num))
                place_in_potentials = 0
        return clues


    '''
    External method to get_next_clue, which gets a list of clues, orders them by similarity, and returns them.
    
    Params:
        game_state: List of ints representing the current state of play
        score: score of the game
        epsilon: optional epsilon param, default will be equal to the score scaled to the range 0-1.
            Intuitively, this means if the score is higher, we will take more risks and put more focus on 
            multi-word clues.
            
    Returns:
        all_clues, a list of Clue type.
    '''
    def get_next_clue(self, board: List[str],
                      allIDs: List[int],
                      game_state: List[int],
                      score: int,
                      epsilon: float=-1.0):

        pos_words = [board[idx] for idx, val in enumerate(allIDs) if val == 1]
        # negative words
        neg_words = [board[idx] for idx, val in enumerate(allIDs) if val != 1]

        available_targets = [word for word in pos_words if game_state[board.index(word)] != -1]
        active_neg_words = [word for word in neg_words if game_state[board.index(word)] != -1]
        logging.info(available_targets)

        #scales epsilon based on score
        if epsilon == -1.0:
            epsilon = (score - -8.0) / (9.0 - -8.0)
        #Gets clues
        all_clues = self._get_clues(available_targets, active_neg_words)

        #sorts by scoring function
        all_clues.sort(key=operator.itemgetter(1))
        all_clues = list(reversed(all_clues))
        all_clues = [clue[0] for clue in all_clues]
        return all_clues[:self.NUM_CLUES]

