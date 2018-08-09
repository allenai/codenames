import torch

class Policy:
    def __init__(self):
        raise NotImplementedError

    '''
    Runs model forward to create a new policy for a given state.
    
    clue_vector: torch Tensor of size d x 1
    options_matrix: torch Tensor of size remaining_words_on_board x size_embed
    
    Returns:
        torch.Tensor of size 1 x remaining_words_on_board.  This is a normalized prob dist of words on 
            game board to choose from.
    '''
    def forward(self,
                clue_vector: torch.Tensor,
                options_matrix: torch.Tensor)-> torch.Tensor:
        raise NotImplementedError