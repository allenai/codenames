import torch as torch
import torch.nn as nn

from codenames.guessers.policy.guesser_policy import GuesserPolicy


class SimilarityThresholdGameStatePolicy(GuesserPolicy, nn.Module):
    '''
    embed_size is the size of the word embeddings
    '''
    def __init__(self, embed_size, seed=42):
        super(GuesserPolicy, self).__init__()

        torch.manual_seed(seed)

        self.embed_size = embed_size
        #Similarity matrix
        w = torch.empty(self.embed_size, self.embed_size)
        nn.init.eye_(w)
        self.W = nn.Parameter(w, requires_grad=True)

        #Similarity threshold, decided by W_t, which is matmul with the game state
        #size 4 x 1 because there are 4 game parameters and 1 threshold param to output.
        self.Wt = nn.Parameter(torch.rand(4,1), requires_grad = True)

    '''
        clue_vector has shape 1 x word_embed_size (represents the embedding of the clue word)
        options_matrix has shape num_words_left_to_guess x word_embed_size (where num_words_left_to_guess is the 
            remaining unguessed words on the game board)
    '''
    def forward(self,
                clue_vector: torch.Tensor,
                options_matrix: torch.Tensor,
                parameterized_game_state: torch.Tensor) -> torch.Tensor:
        m = nn.Sigmoid()
        predicted_similarities = m(torch.matmul(torch.matmul(clue_vector, self.W),
                                                torch.t(options_matrix)))
        calculated_threshold = m(torch.matmul(parameterized_game_state, self.Wt))
        clamped_similarities = torch.clamp(predicted_similarities - calculated_threshold, 0.0, 1.0)
        return clamped_similarities


#Main method included only for testing purposes
def main():
    clue_vect = torch.empty(1,50)
    options_vect = torch.empty(10,50)
    game_state = torch.empty(1, 4)
    policy = SimilarityThresholdGameStatePolicy(50)
    policy.forward(clue_vect, options_vect, game_state)

if __name__ == "__main__":
    main()
