import torch


class GuesserPolicy:
    def __init__(self):
        raise NotImplementedError

    '''
    Runs model forward to create a new policy for a given state.
    
    Inputs should be specified by child instantiations.`
    '''

    def forward(self) -> torch.Tensor:
        raise NotImplementedError
