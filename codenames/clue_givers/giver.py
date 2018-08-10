from typing import List
from codenames.utils.game_utils import Clue

class Giver:
    """
    Parameters
    ----------
    board : `List[str]`
        List of all words on the board in the current game
    allIDs: `List[int]`
        List of assignment codes for all words on the board
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
            List of same size as self.board telling whether a word on board is already revealed
        score : `int`
            Current score
        """
        raise NotImplementedError
