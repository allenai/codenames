import torch as torch
import torch.nn as nn

from codenames.guessers.policy.guesser_policy import GuesserPolicy


class SimilarityThresholdPolicy(GuesserPolicy, nn.Module):
    '''
    embed_size is the size of the word embeddings
    '''
    def __init__(self, embed_size):
        super(GuesserPolicy, self).__init__()
        self.embed_size = embed_size
        #Similarity matrix
        self.W = nn.Parameter(torch.rand(self.embed_size, self.embed_size), requires_grad = True)

        #Similarity threshold
        self.threshold = nn.Parameter(torch.Tensor([0.01]), requires_grad = True)

    '''
        clue_vector has shape 1 x word_embed_size (represents the embedding of the clue word)
        options_matrix has shape num_words_left_to_guess x word_embed_size (where num_words_left_to_guess is the 
            remaining unguessed words on the game board)
    '''
    def forward(self,
                clue_vector: torch.Tensor,
                options_matrix: torch.Tensor) -> torch.Tensor:
        m = nn.Sigmoid()
        predicted_similarities = m(torch.matmul(torch.matmul(clue_vector, self.W),
                                                torch.t(options_matrix)))
        clamped_similarities = torch.clamp(predicted_similarities - self.threshold, 0.0, 1.0)
        return clamped_similarities


#Main method included only for testing purposes
def main():
    clue_vect = torch.empty(1,50)
    options_vect = torch.empty(10,50)
    policy = SimilarityThresholdPolicy(50)
    policy.forward(clue_vect, options_vect)

if __name__ == "__main__":
    main()
