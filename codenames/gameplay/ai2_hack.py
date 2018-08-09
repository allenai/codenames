import sys

from random import choices, shuffle
from typing import List
from termcolor import colored

from codenames.clue_givers.giver import Giver, Clue
from codenames.embedding_handler import EmbeddingHandler
from codenames.guessers.guesser import Guesser
from codenames.utils.game_utils import UNREVEALED, ASSASSIN, GOOD, BAD
from codenames.gameplay.engine import GameEngine

SCORE_CORRECT_GUESS = 1
SCORE_INCORRECT_GUESS = -1
SCORE_ASSASSIN_GUESS = -1
SCORE_CIVILIAN_GUESS = 0


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

        self.result = None

    def is_game_over(self):
        team1_has_words_left_to_guess, team2_has_words_left_to_guess = False, False
        for i in range(len(self.engine.owner)):
            # if the assassin is revealed, then it's game over.
            if self.engine.owner[i] == ASSASSIN and not self.engine.assignment_not_revealed[i]:
                self.result = "Assassin word guessed"
                return True
            # does team1/2 has any invisible words?
            if self.engine.owner[i] == GOOD and self.engine.assignment_not_revealed[i]:
                team1_has_words_left_to_guess = True
            if self.engine.owner[i] == BAD and self.engine.assignment_not_revealed[i]:
                team2_has_words_left_to_guess = True

        # if all words of either team are visible, it's game over.
        if not team1_has_words_left_to_guess:
            self.result = "Team 1: Winner"
            return True
        if not team2_has_words_left_to_guess:
            self.result = "Team 2: Winner"
            return True

        # if none of the above conditions apply, the game is not over.
        return False

    def is_valid_clue(self, clue_word: str):
        return True

    def _apply_team1_guess(self, guess):
        idx = self.engine.board.tolist().index(guess)
        self.engine.assignment_not_revealed[idx] = False

        if idx == -1:
            raise Exception
        else:
            self.game_state[idx] = self.engine.owner[idx]
            if self.engine.owner[idx] == GOOD:
                guess_reward = SCORE_CORRECT_GUESS
            elif self.engine.owner[idx] == BAD:
                guess_reward = SCORE_INCORRECT_GUESS
            elif self.engine.owner[idx] == ASSASSIN:
                guess_reward = SCORE_ASSASSIN_GUESS
            else:
                guess_reward = SCORE_CIVILIAN_GUESS

            self.cumulative_score += guess_reward
            return guess_reward

    # This method executes the guesses of team1.
    def apply_team1_guesses(self, clue: Clue, guessed_words: List[str]):

        guess_list_rewards = []
        if len(guessed_words) > int(clue.count) + 1:
            raise Exception
        for word in guessed_words:
            guess_reward = self._apply_team1_guess(word)
            guess_list_rewards.append(guess_reward)
            if guess_reward <= 0:
                break

        # team1 just played, so we need to update the internal state of
        # the engine pertaining to turn info.
        self.engine.next_turn()

        return guess_list_rewards

    # The engine was designed for a 2-player experience. In the 1-player
    # version of the game, there is an imaginary team2 which is pretty
    # boring: it reveals one of the team2 cards in each turn. This method
    # simulates the imaginary boring team2 for the 1-player version of the 
    # game.
    def apply_team2_guesses(self):
        team2_guess = None
        for i in range(len(self.engine.board)):
            # find the first word which belongs to team2 and is not revealed.
            if self.engine.owner[i] == BAD and self.engine.assignment_not_revealed[i]:
                # then reveal it.
                self.game_state[i] = self.engine.owner[i]
                self.engine.assignment_not_revealed[i] = False
                team2_guess = self.engine.board[i]
                break

        # This should never happen.
        if not team2_guess:
            raise RuntimeError('All cards which belong to TEAM2 have already'
                               'been revealed, why are we still playing?')

        # team2 just played, so we need to update the internal state of
        # the engine pertaining to turn info.
        self.engine.next_turn()

        return [team2_guess]

class RandomGiver(Giver):
    '''
  A clue giver who randomly picks the clue word from a vocabulary.
  '''

    def __init__(self, board: List[str], target_IDs: List[int],
                 vocab=None):
        super().__init__(board, target_IDs)
        if vocab is None:
            vocab = ['I', 'have', 'no', 'clue', 'what', 'I', 'am', 'doing']
        self.vocab = vocab

    def get_next_clue(self, game_state, score, black_list=None):
        options = choices(self.vocab, k=3)
        clues = []
        for option in options:
            clues.append(Clue(clue_word=option, intended_board_words=[], count=1))
        return clues


class RandomGuesser(Guesser):
    '''
  A guesser who randomly picks among unrevealed board words.
  '''

    def __init__(self, board: List[str]):
        super().__init__(board)
        self.board = board

    def guess(self, clue_word, count, game_state, cumulative_score):
        unrevealed_words = []
        for i in range(len(game_state)):
            if game_state[i] == -1:
                unrevealed_words.append(self.board[i])
        return choices(unrevealed_words, k=count + 1)

    def report_reward(self, reward):
        pass


def play_game(board_size=5, giver_options=[], guesser_options=[], board_data=None):
    print('||| initializing all modules.')
    game = GameWrapper(board_size, board_data)
    giver = RandomGiver(game.engine.board, game.engine.owner)
    guesser = RandomGuesser(game.engine.board)

    print('||| data: {}.'.format(list(zip(game.engine.board, game.engine.owner))))

    turn = 1
    while not game.is_game_over():
        if turn == 1: 
            game.engine.print_board(spymaster=True)
        input('||| press ENTER to see the next clue for team1.')

        # get a list of clues.
        clue_objects = giver.get_next_clue(game.game_state, game.cumulative_score)
        # find the first legal clue, then proceed.
        first_valid_clue = None
        for clue in clue_objects:
            if game.is_valid_clue(clue_objects[0].clue_word):
                first_valid_clue = clue
                break
                
        if first_valid_clue is None:
            raise RuntimeError('All clues given were illegal.')

        clue_word, clue_count = first_valid_clue.clue_word, first_valid_clue.count
        # get guesses.
        sys.stdout.write("||| team1's clue: ({}, {}).\t".format(clue.clue_word, clue.count))
        guessed_words = guesser.guess(clue_word, clue_count, game.game_state, game.cumulative_score)
        input(', press ENTER to see team1 guesses.')

        guess_list_rewards = game.apply_team1_guesses(first_valid_clue, guessed_words)
        guesser.report_reward(guess_list_rewards)

        # print the board after team1 plays this turn.
        game.engine.print_board(spymaster=True)
        sys.stdout.write("||| team1's clue: ({}, {}).\n".format(clue.clue_word, clue.count))
        print("||| team1's guess: {}".format(guessed_words))
        sys.stdout.write('||| rewards: {}\t'.format(list(zip(guessed_words, guess_list_rewards))))
        if not game.is_game_over():
            input(", press ENTER to see team2's next move.")
            team2_guessed_words = game.apply_team2_guesses()
            # print the board again after team2 plays this turn.
            game.engine.print_board(spymaster=True)
            sys.stdout.write("||| team1's clue: ({}, {}).\n".format(clue.clue_word, clue.count))
            print("||| team1's guess: {}".format(guessed_words))
            sys.stdout.write('||| rewards: {}\n'.format(list(zip(guessed_words, guess_list_rewards))))
            sys.stdout.write("||| team2 revealed: {}\n".format(team2_guessed_words))

        turn += 1

    # experitment with white background, and darker civilians.
    # display the guesses that were actually played.
    print('\n||| termination condition: {}'.format(game.result))
    print('|||')
    print('||| =============== GAME OVER =================')
    print('||| =============== team1 score: {}'.format(game.cumulative_score))

def main():
    play_game()
  
if __name__== "__main__":
    main()
