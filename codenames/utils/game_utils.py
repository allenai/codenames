from typing import List


UNREVEALED = -1

GOOD = 0

BAD = 1

CIVILIAN = 2

ASSASSIN = 3


def get_available_choices(board: List[str],
                          game_state: List[int]) -> List[str]:
    assert len(board) == len(game_state), "Invalid state!"
    options = []
    for string, state_id in zip(board, game_state):
        if state_id == UNREVEALED:
            options.append(string)
    return options
