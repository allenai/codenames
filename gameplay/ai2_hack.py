from collections import namedtuple
from random import choice

from gameplay.engine import GameEngine


class GameWrapper():

  def __init__(self, board_size, board_data = None):
    '''
    board_size: int, number of board_words = board_size * board_size
    board_data: string, format as: ASSASSIN;TEAM1;TEAM2;NEUTRAL
    where each group consists of comma-separated words from the word list.
    '''
    engine = GameEngine()
    # initialize board data.
    if board_data == None:
      engine.initialize_random_game(size = board_size)
    else:
      engine.initialize_initialize_from_words(board_data, size = board_size)

    # initialize game state.
    self.game_state = [-1] * (board_size * board_size)

    # initialize score.
    self.score = 0

  def is_game_over(self):
    team1_has_invisible_words, team2_has_invisible_words = False, False
    for i in range(self.engine.assignments):
      # if the assassin is revealed, then it's game over.
      if self.engine.assignments[i] == 0 and self.engine.visible[i]:
        return True
      # does team1/2 has any invisible words?
      if self.engine.assignments[i] == 1 and not self.engine.visible[i]:
        team1_has_invisible_words = True
      if self.engine.assignments[i] == 2 and not self.engine.visible[i]:
        team2_has_invisible_words = True

    # if all words of either team are visible, it's game over.
    if not team1_has_invisible_words:
      return True
    if not team2_has_invisible_words:
      return True

    # if none of the above conditions apply, the game is not over.
    return False

  def is_valid_clue(self, clue_word: str):
    return True

  def apply_guesses(self, clue: Clue, guessed_words: List[str]):
    raise NotImplementedError

Clue = namedtuple('Clue', 'clue_word', 'intended_board_words', 'count')

class RandomGiver():
  '''
  A clue giver who randomly picks the clue word from a vocabulary.
  '''
  def __init__(self, vocab = ['I', 'have', 'no', 'clue', 'what', 'I', 'am', 'doing']):
    self.vocab = vocab

  def get_next_clue(self, game_state, score, black_list):
    return choice(self.vocab)

class RandomGuesser():
  '''
  A guesser who randomly picks among unrevealed board words.
  '''

  def __init__(self, board_words):
    self.board_words = board_words

  def guess(self, clue_word, game_state, count):
    unrevealed_words = []
    for i in range(game_state):
      if game_state[i] == -1:
        unrevealed_words.append(self.board_words[i])
    return choice(unrevealed_words)

def play_game(board_size = 5, giver_options = [], guesser_options = [], board_data = None):
  game = GameWrapper(board_size, board_data)
  giver = RandomGiver()
  guesser = RandomGuesser(board_words)

  while not game.is_game_over():
    # get a list of clues.
    clue_objects = giver.get_next_clue(game_state, score)
    # find the first legal clue.
    while len(clue_objects) > 0:
      if not game.is_valid_clue(clue_objects[0].clue_word):
        del clue_objects[0]
    if len(clue_objects) == 0:
      raise RuntimeError('All clues given were illegal.')
    clue_word, clue_count  = clue_objects[0].clue_word, clue_objects[0].count
    # get guesses.
    guessed_words = guesser.guess(clue_word, clue_count, game.game_state, game.score)
    game.apply_guesses(clue_objects[0], guessed_words)
