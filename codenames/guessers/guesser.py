from typing import List


class Guesser:
    """
    Parameters
    ----------
    board : `List[str]`
        List of all words on the board in the current game
    embedding_file : `str`
        Location of pickled embeddings
    """
    def guess(self,
              board: List[str],
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

    def report_reward(self, rewards: List[int]) -> None:
        pass
