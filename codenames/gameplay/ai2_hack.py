import sys
import os

from random import choices, shuffle
from typing import List
from termcolor import colored
import argparse
import torch

from codenames.clue_givers.giver import Giver, Clue
from codenames.clue_givers.heuristic_giver import HeuristicGiver
from codenames.embedding_handler import EmbeddingHandler
from codenames.guessers.guesser import Guesser
from codenames.guessers.heuristic_guesser import HeuristicGuesser
from codenames.guessers.learned_guesser import LearnedGuesser
from codenames.guessers.policy.similarity_threshold import SimilarityThresholdPolicy
from codenames.utils.game_utils import UNREVEALED, ASSASSIN, GOOD, BAD, Clue
from codenames.gameplay.engine import GameEngine

SCORE_CORRECT_GUESS = 1
SCORE_INCORRECT_GUESS = -1
SCORE_ASSASSIN_GUESS = -5
SCORE_CIVILIAN_GUESS = -1


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

        return guess_list_rewards + [0] * (len(guessed_words) - len(guess_list_rewards))

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
    def __init__(self, embedding_handler):
        self.vocab = list(embedding_handler.word_indices.keys())

    def get_next_clue(self,
                      board,
                      allIDs,
                      game_state,
                      score):
        options = choices(self.vocab, k=3)
        clues = []
        for option in options:
            clues.append(Clue(clue_word=option, intended_board_words=[], count=1))
        return clues


class RandomGuesser(Guesser):
    '''
    A guesser who randomly picks among unrevealed board words.
    '''
    def guess(self, board, clue_word, count, game_state, cumulative_score):
        unrevealed_words = []
        for i in range(len(game_state)):
            if game_state[i] == -1:
                unrevealed_words.append(board[i])
        return choices(unrevealed_words, k=count + 1)

    def report_reward(self, reward):
        pass

def _print(message, verbose):
    if verbose:
        sys.stdout.write(message)

def _input(message, verbose):
    if verbose:
        return input(message)
    else:
        return ''

def play_game(giver, guesser, board_size=5, board_data=None, verbose=True, saved_path=None):
    _print('||| initializing all modules.\n', verbose=verbose)
    game = GameWrapper(board_size, board_data)
    _print('||| data: {}.\n'.format(list(zip(game.engine.board, game.engine.owner))), verbose=verbose)

    turn = 1
    while not game.is_game_over():
        if turn == 1: 
            game.engine.print_board(spymaster=True, verbose=verbose)
        _input('\n||| press ENTER to see the next clue for team1.', verbose=verbose)

        # get a list of clues.
        board = game.engine.board.tolist()
        clue_objects = giver.get_next_clue(board,
                                           game.engine.owner,
                                           game.game_state,
                                           game.cumulative_score)
        # find the first legal clue, then proceed.
        first_valid_clue = None
        for clue in clue_objects:
            if game.is_valid_clue(clue_objects[0].clue_word):
                first_valid_clue = clue
                break

        if first_valid_clue is None:
            # All clues are illegal. Abandoning game!
            break

        clue_word, clue_count = first_valid_clue.clue_word, first_valid_clue.count
        # get guesses.
        _print("||| team1's clue: ({}, {}).\n".format(clue.clue_word, clue.count), verbose=verbose)
        _print("||| \tIntended target words: [{}]\n".format(', '.join(clue.intended_board_words)),
               verbose=verbose)

        guessed_words = guesser.guess(game.engine.board,
                                      clue_word,
                                      clue_count,
                                      game.game_state,
                                      game.cumulative_score)
        _input(', press ENTER to see team1 guesses.\n', verbose=verbose)

        guess_list_rewards = game.apply_team1_guesses(first_valid_clue, guessed_words)
        _print('||| rewards: {}'.format(list(zip(guessed_words, guess_list_rewards))),
               verbose=verbose)
        if saved_path and game.is_game_over():
            guesser.report_reward(guess_list_rewards, saved_path)
        else:
            guesser.report_reward(guess_list_rewards)
        turn += 1

    _print('\n||| termination condition: {}\n'.format(game.result), verbose=verbose)
    _print('|||\n', verbose=verbose)
    _print('||| =============== GAME OVER =================\n', verbose=verbose)
    _print('||| =============== team1 score: {}\n'.format(game.cumulative_score), verbose=verbose)
    # If game.result is None, it means that the giver could not give a clue, and the game was
    # abandoned.
    score = game.cumulative_score if game.result is not None else 0
    return score

def main(args):
    embedding_handler = EmbeddingHandler('data/uk_embeddings.txt')
    if args.giver_type == "heuristic":
        giver = HeuristicGiver(embedding_handler)
    elif args.giver_type == "random":
        giver = RandomGiver(embedding_handler)
    else:
        raise NotImplementedError

    if args.guesser_type == "heuristic":
        guesser = HeuristicGuesser(embedding_handler)
    elif args.guesser_type == "random":
        guesser = RandomGuesser()
    elif args.guesser_type == "learned":
        if args.load_model:
            guesser = LearnedGuesser(embedding_handler,
                                     policy=torch.load(args.load_model),
                                     learning_rate=0.01)
        else:
            guesser = LearnedGuesser(embedding_handler,
                                     policy=SimilarityThresholdPolicy(300),
                                     learning_rate=0.01)
    else:
        raise NotImplementedError
    if args.interactive:
        play_game(giver=giver, guesser=guesser,
                  board_size=args.board_size, verbose=True)
    else:
        scores = []
        num_wins = 0
        for i in range(args.num_games):
            saved_path = ""
            if args.guesser_type == "learned" and (i % 100 == 0 or i == args.num_games - 1):
                if not os.path.exists("./models"):
                    os.makedirs("./models")
                saved_path = "./models/learned" + str(i)

            score = play_game(giver=giver, guesser=guesser,
                              board_size=args.board_size, verbose=False,
                              saved_path=saved_path)
            if score > 0:
                num_wins += 1
            scores.append(score)

        mean_score = sum(scores)/len(scores)
        print(f"Played {args.num_games} games.")
        print(f"Team 1 won {num_wins} times.")
        print(f"Average score is {mean_score}")


if __name__== "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--guesser", type=str, dest="guesser_type", default="heuristic")
    argparser.add_argument("--giver", type=str, dest="giver_type", default="heuristic")
    argparser.add_argument("--size", type=int, dest="board_size", default="5")
    argparser.add_argument("--interactive", action="store_true")
    argparser.add_argument("--num-games", type=int, dest="num_games",
                           help="Number of games to play if not interactive (default=1000)",
                           default=1000)
    argparser.add_argument("--load-model", dest="load_model", default=None)
    args = argparser.parse_args()
    main(args)
