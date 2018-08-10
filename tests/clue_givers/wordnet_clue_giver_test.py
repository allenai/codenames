
from unittest import TestCase

from codenames.clue_givers.wordnet_cluegiver import WordnetClueGiver
from codenames.utils.game_utils import Clue


class TestWordnetClueGiver(TestCase):
    def test_clues(self):
        test_board = ["woman", "man", "girl", "boy", "blue", "cat", "queen", "king"]
        test_allIDs = [1, 2, 1, 2, -1, -1, 1, 2]
        cg = WordnetClueGiver()
        clues = cg.get_next_clue(test_board, test_allIDs, [1, 2, 1, 2, -1, -1, 1, 2], 3)
        assert clues[0] == Clue(clue_word='female', intended_board_words=('girl', 'woman'), count=2)




