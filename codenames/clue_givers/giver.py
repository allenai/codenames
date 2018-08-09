from typing import List
from collections import namedtuple
from codenames.embedding_handler import EmbeddingHandler
#from codenames.util import UNREVEALED, GOOD, BAD, CIVILIAN, ASSASIN
#from codenames.gameplay.ai2_hack import Clue

from codenames.utils.game_utils import Clue


# from codenames.util import UNREVEALED, GOOD, BAD, CIVILIAN, ASSASIN
#from codenames.gameplay.ai2_hack import Clue

class Giver:
    """
    Parameters
    ----------
    board : `List[str]`
        List of all words on the board in the current game
    target_IDs: `List[str]`
        List of all target words the Giver wants the Guesser to guess correctly
    embedding_file : `str`
        Location of pickled embeddings
    """

    def __init__(self,
                 board: List[str],
                 allIDs: List[int]) -> None:
        self.board = board
        self.allIDs = allIDs

    def get_next_clue(self,
                      game_state: List[int],
                      current_score: int) -> List[Clue]:
        """
        Parameters
        ----------
        game_state : `List[int]`
            List of same size as self.board, with each element showing the ids revealed so far (eg.
            same team, opposite team, assasin, civilian etc.)
        score : `int`
            Current score
        """
        raise NotImplementedError
