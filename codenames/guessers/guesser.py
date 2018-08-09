from typing import List

from codenames.embedding_handler import EmbeddingHandler
from codenames.utils.game_utils import UNREVEALED


class Guesser:
    """
    Parameters
    ----------
    board : `List[str]`
        List of all words on the board in the current game
    embedding_file : `str`
        Location of pickled embeddings
    """
    def __init__(self,
                 board: List[str],
                 embedding_handler: EmbeddingHandler=None) -> None:
        self.board = board
        self.embedding_handler = embedding_handler

    def guess(self,
              clue: str,
              count: int,
              game_state: List[int],
              current_score: int) -> List[str]:
        """
        Parameters
        ----------
        clue : `str`
        count : `int`
            Max size of returned list
        game_state : `List[int]`
            List of same size as self.board, with each element showing the ids revealed so far (eg.
            same team, opposite team, assasin, civilian etc.)
        score : `int`
            Current score
        """
        raise NotImplementedError
