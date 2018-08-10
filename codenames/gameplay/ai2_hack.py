import sys
import os
import re
import tqdm
import datetime
from collections import defaultdict

from random import choices, shuffle
from typing import List
from termcolor import colored
import numpy as np
import argparse
import torch

from codenames.clue_givers.giver import Giver, Clue
from codenames.clue_givers.heuristic_giver import HeuristicGiver
from codenames.clue_givers.wordnet_cluegiver import WordnetClueGiver
from codenames.embedding_handler import EmbeddingHandler
from codenames.guessers.guesser import Guesser
from codenames.guessers.heuristic_guesser import HeuristicGuesser
from codenames.guessers.learned_guesser import LearnedGuesser
from codenames.guessers.policy.similarity_threshold_game_state import SimilarityThresholdGameStatePolicy
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
            self.engine.initialize_from_words(board_data, size=board_size)

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
    '''
    returns cumulative_team1_score: int, termination_condition: str, team1_turns: int
    '''
    _print('||| initializing all modules.\n', verbose=verbose)
    game = GameWrapper(board_size, board_data)
    _print('||| data: {}.\n'.format(list(zip(game.engine.board, game.engine.owner))),
           verbose=verbose)

    turn = 0
    while not game.is_game_over():
        if turn == 0:
            game.engine.print_board(spymaster=True, verbose=verbose)
        else:
            game.engine.print_board(spymaster=True, verbose=verbose, clear_screen=False)
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
            game.result = 'All clues given were illegal.'
            break

        clue_word, clue_count = first_valid_clue.clue_word, first_valid_clue.count
        # get guesses.
        _print("||| team1's clue: ({}, {}); \tIntended target words: [{}]\n".format(clue.clue_word, clue.count, clue.intended_board_words), verbose=verbose)

        guessed_words = guesser.guess(game.engine.board,
                                      clue_word,
                                      clue_count,
                                      game.game_state,
                                      game.cumulative_score)
        _input(', press ENTER to see team1 guesses.\n', verbose=verbose)

        guess_list_rewards = game.apply_team1_guesses(first_valid_clue, guessed_words)

        rewards_out = []
        for w, r in zip(guessed_words, guess_list_rewards):
            if r < SCORE_CORRECT_GUESS:
                break
            rewards_out.append((w, r))
        _print('||| rewards: {}\n'.format(rewards_out), verbose=verbose)

        if saved_path and game.is_game_over():
            guesser.report_reward(guess_list_rewards, saved_path)
        else:
            guesser.report_reward(guess_list_rewards)

        # print the board after team1 plays this turn.
        game.engine.print_board(spymaster=True, verbose=verbose)
        _print("||| team1's clue: ({}, {}); \tIntended target words: [{}]\n".format(clue.clue_word, clue.count, clue.intended_board_words), verbose=verbose)
        _print("||| team1's guesses: {}\n".format(list(zip(guessed_words, guess_list_rewards))), verbose=verbose)
        if not game.is_game_over():
            _input(", press ENTER to see team2's next move.", verbose=verbose)
            team2_guessed_words = game.apply_team2_guesses()
            # print the board again after team2 plays this turn.
            game.engine.print_board(spymaster=True, verbose=verbose)
            _print("||| team1's clue: ({}, {}).\n".format(clue.clue_word, clue.count), verbose=verbose)
            _print("||| team1's guess: {}\n".format(list(zip(guessed_words, guess_list_rewards))), verbose=verbose)
            _print("||| team2 revealed: {}\n".format(team2_guessed_words), verbose=verbose)

        turn += 1

    _print('\n||| termination condition: {}\n'.format(game.result), verbose=verbose)
    _print('|||\n', verbose=verbose)
    _print('||| =============== GAME OVER =================\n', verbose=verbose)
    _print('||| =============== team1 score: {}\n'.format(game.cumulative_score), verbose=verbose)
    assert game.result is not None
    return game.cumulative_score, game.result, turn
    

def main(args):
    guesser_embedding_handler = EmbeddingHandler(args.guesser_embeddings_file)
    giver_embedding_handler = EmbeddingHandler(args.giver_embeddings_file)
    if args.giver_type == "heuristic":
        giver = HeuristicGiver(giver_embedding_handler)
    elif args.giver_type == "random":
        giver = RandomGiver(giver_embedding_handler)
    elif args.giver_type == "wordnet":
        giver = WordnetClueGiver()
    else:
        raise NotImplementedError

    if args.game_data:
        all_game_data = []
        for line in open(args.game_data, mode='rt'):
            # allow comments starting with # or %
            if line.startswith("#"): continue
            if line.startswith("%"): continue
            line = line.strip()
            words = re.split('[;,]', line)
            if len(words) != args.board_size * args.board_size:
                if args.verbose:
                    sys.stdout.write('WARNING: skipping game data |||{}||| due to a conflict with the specified board size: {}'.format(line, args.board_size))
                continue
            all_game_data.append(line.strip())
            shuffle(all_game_data)
            all_game_data = all_game_data[:args.num_games]
    else:
        # If game data were not specified, we'd like to generate (args.num_games) random 
        # games. The method `play_game` randomly samples words when the provided game data
        # is set to None.
        all_game_data = [None] * args.num_games

    if args.guesser_type == "heuristic":
        guesser = HeuristicGuesser(guesser_embedding_handler)
    elif args.guesser_type == "random":
        guesser = RandomGuesser()
    elif args.guesser_type == "learned":
        if args.load_model:
            guesser = LearnedGuesser(guesser_embedding_handler,
                                     policy=torch.load(args.load_model),
                                     learning_rate=0.01,
                                     train=False)
        else:
            guesser = LearnedGuesser(guesser_embedding_handler,
                                     policy=SimilarityThresholdPolicy(300),
                                     learning_rate=0.01,
                                     train=True)
    elif args.guesser_type == "learnedstate":
        if args.load_model:
            guesser = LearnedGuesser(guesser_embedding_handler,
                                     policy=torch.load(args.load_model),
                                     learning_rate=0.01,
                                     train=False)
        else:
            guesser = LearnedGuesser(guesser_embedding_handler,
                                     policy=SimilarityThresholdGameStatePolicy(300),
                                     learning_rate=0.01,
                                     train=True)
    else:
        raise NotImplementedError
    
    # keep track of the results of each game.
    all_scores = []
    all_termination_conditions = defaultdict(int)
    all_turns = []
    num_positive_score = 0
    start_time = datetime.datetime.now()
    for i, board_data in tqdm.tqdm(enumerate(all_game_data), desc="games played: "):
        saved_path = ""
        save_now = i % 100
        if args.num_games is not None:
            save_now = save_now or i == args.num_games - 1
        if args.guesser_type == "learned" and save_now:
            if not os.path.exists("./models"):
                os.makedirs("./models")
            saved_path = "./models/learned" + str(i)

        print(saved_path)
        score, termination_condition, turns = play_game(giver=giver, guesser=guesser,
                                                        board_size=args.board_size, 
                                                        board_data=board_data,
                                                        verbose=args.interactive,
                                                        saved_path=saved_path)
        if score > 0:
            num_positive_score += 1
        all_scores.append(score)
        all_termination_conditions[termination_condition] += 1
        all_turns.append(turns)
        mean_score = sum(all_scores) / len(all_scores)
        std_score = np.std(all_scores)
        mean_turns = sum(all_turns) / len(all_turns)
        std_turns = np.std(all_turns)

        # log, for debugging purposes.
        if args.verbose:
            # this game's results
            sys.stdout.write('||| last game score = {}, termination condition = {}, turns = {}\n'.format(score, termination_condition, turns))
            # summary of all games' results
            sys.stdout.write('|||\n')
            sys.stdout.write("||| # of games played = {}, runtime = {}\n".format(len(all_scores), str(datetime.datetime.now() - start_time)))
            sys.stdout.write(f"||| # of games won (by team1) = {num_positive_score}\n")
            sys.stdout.write('||| avg. game score = {:.2f}, std. of game score = {:.2f}\n'.format(mean_score, std_score))
            sys.stdout.write('||| avg. game turns = {:.2f}, std. of game turns = {:.2f}\n'.format(mean_turns, std_turns))
            for _termination_condition, _count in all_termination_conditions.items():
                sys.stdout.write('||| % of {}: {:.2f}\n'.format(_termination_condition, 1.0 * _count / len(all_scores)))

    with open(args.experiment_name + '.experiment', mode='wt') as experiment_results_file:
        experiment_results_file.write('name: {}\n'.format(args.experiment_name))
        experiment_results_file.write('runtime: {}\n'.format(str(datetime.datetime.now() - start_time)))
        experiment_results_file.write('time finished: {}\n'.format(str(datetime.datetime.now())))
        experiment_results_file.write("# of games played: {}\n".format(len(all_scores)))
        experiment_results_file.write(f"# of games won (by team1) = {num_positive_score}\n")
        for _termination_condition, _count in all_termination_conditions.items():
            experiment_results_file.write('% of {}: {:.2f}\n'.format(_termination_condition, 1.0 * _count / len(all_scores)))
        experiment_results_file.write('avg. game turns = {:.2f}, std. of game turns = {:.2f}\n'.format(mean_turns, std_turns))
        experiment_results_file.write('avg. game score = {:.2f}, std. of game score = {:.2f}\n'.format(mean_score, std_score))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--guesser", type=str, dest="guesser_type", default="heuristic")
    argparser.add_argument("--giver", type=str, dest="giver_type", default="heuristic")
    argparser.add_argument("--size", type=int, dest="board_size", default="5")
    argparser.add_argument("--interactive", action="store_true")
    argparser.add_argument("--num-games", type=int, help="Number of games to play")
    argparser.add_argument("--game-data", type=str, default="data/codenames_dev.games")
    argparser.add_argument("--guesser-embeddings-file", type=str, dest="guesser_embeddings_file",
                           default="data/uk_embeddings.txt")
    argparser.add_argument("--giver-embeddings-file", type=str, dest="giver_embeddings_file",
                           default="data/uk_embeddings.txt")
    argparser.add_argument("--load-model", dest="load_model", default=None,
                           help="Will not train if this argument is set.")
    argparser.add_argument("--experiment-name", type=str, default="debug")
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()
    main(args)
