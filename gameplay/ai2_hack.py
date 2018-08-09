from collections import namedtuple
from random import choice, shuffle
from typing import List

from codenames.utils.game_utils import UNREVEALED, ASSASSIN, GOOD, BAD
from gameplay.engine import GameEngine

Clue = namedtuple('Clue', ['clue_word', 'intended_board_words', 'count'])

SCORE_CORRECT_GUESS = 1
SCORE_INCORRECT_GUESS = -1
SCORE_ASSASSIN_GUESS = -1

class GameWrapper:

    def __init__(self, board_size, board_data=None):
        '''
    board_size: int, number of board_words = board_size * board_size
    board_data: string, format as: ASSASSIN;TEAM1;TEAM2;NEUTRAL
    where each group consists of comma-separated words from the word list.
    '''
        self.engine = GameEngine()
        # initialize board data.
        if board_data == None:
            self.engine.initialize_random_game(size=board_size)
        else:
            self.engine.initialize_initialize_from_words(board_data, size=board_size)

        # initialize game state.
        self.game_state = [UNREVEALED] * (board_size * board_size)

        # initialize score.
        self.cumulative_score = 0

    def is_game_over(self):
        team1_has_words_left_to_guess, team2_has_words_left_to_guess = False, False
        for i in range(self.engine.owner):
            # if the assassin is revealed, then it's game over.
            if self.engine.owner[i] == ASSASSIN and not self.engine.assignment_not_revealed[i]:
                return True
            # does team1/2 has any invisible words?
            if self.engine.owner[i] == GOOD and self.engine.assignment_not_revealed[i]:
                team1_has_words_left_to_guess = True
            if self.engine.owner[i] == BAD and not self.engine.assignment_not_revealed[i]:
                team2_has_words_left_to_guess = True

        # if all words of either team are visible, it's game over.
        if not team1_has_words_left_to_guess:
            return True
        if not team2_has_words_left_to_guess:
            return True

        # if none of the above conditions apply, the game is not over.
        return False

    def is_valid_clue(self, clue_word: str):
        return True

    def _apply_guess(self, guess):
        idx = self.engine.board.indexof(guess)
        self.engine.assignment_not_revealed[idx] = False

        if idx == -1:
            raise Exception
        else:
            turn_reward = None
            self.game_state[idx] = self.engine.owner[idx]
            if self.engine.owner[idx] == GOOD:
                turn_reward = SCORE_CORRECT_GUESS
            elif self.engine.owner[idx] == BAD:
                turn_reward = SCORE_INCORRECT_GUESS
            elif self.engine.owner[idx] == ASSASSIN:
                turn_reward = SCORE_ASSASSIN_GUESS

            self.cumulative_score += turn_reward
            return turn_reward

    def apply_guesses(self, clue: Clue, guessed_words: List[str]):
        turn_reward = 0
        if len(guessed_words) > int(clue.count) + 1:
            raise Exception
        for word in guessed_words:
            turn_reward += self._apply_guess(word)

        return turn_reward


class RandomGiver:
    '''
  A clue giver who randomly picks the clue word from a vocabulary.
  '''

    def __init__(self, vocab=['I', 'have', 'no', 'clue', 'what', 'I', 'am', 'doing']):
        self.vocab = vocab

    def get_next_clue(self, game_state, score, black_list):
        return choice(self.vocab)


class RandomGuesser:
    '''
  A guesser who randomly picks among unrevealed board words.
  '''

    def __init__(self, board_words):
        self.board_words = board_words

    def guess(self, clue_word, count, game_state, cumulative_score):
        unrevealed_words = []
        for i in range(game_state):
            if game_state[i] == -1:
                unrevealed_words.append(self.board_words[i])
        return shuffle(unrevealed_words)[:count+1]

    def report_reward(self, reward):
        pass


def play_game(board_size=5, giver_options=[], guesser_options=[], board_data=None):
    game = GameWrapper(board_size, board_data)
    giver = RandomGiver()
    guesser = RandomGuesser(game.engine.board)

    while not game.is_game_over():
        # get a list of clues.
        clue_objects = giver.get_next_clue(game.game_state, game.cumulative_score)
        # find the first legal clue.
        while len(clue_objects) > 0:
            if not game.is_valid_clue(clue_objects[0].clue_word):
                del clue_objects[0]
        if len(clue_objects) == 0:
            raise RuntimeError('All clues given were illegal.')
        clue_word, clue_count = clue_objects[0].clue_word, clue_objects[0].count
        # get guesses.
        guessed_words = guesser.guess(clue_word, clue_count, game.game_state, game.cumulative_score)
        turn_reward = game.apply_guesses(clue_objects[0], guessed_words)
        guesser.report_reward(turn_reward)
