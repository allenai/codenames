from typing import List
from codenames.utils.game_utils import Clue


class Giver:
    def get_next_clue(self,
                      board: List[str],
                      allIDs: List[int],
                      game_state: List[int],
                      current_score: int) -> List[Clue]:
        """
        Parameters
        ----------
        board : `List[str]`
            All the words on the board.
        allIDs : `List[int]`
            The true identities of all the words
        game_state : `List[int]`
            List of same size as self.board telling whether a word on board is already revealed
        score : `int`
            Current score
        """
        raise NotImplementedError
