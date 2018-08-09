from policy import Policy
import torch as torch
import torch.nn as nn

class SimilarityThresholdPolicy(Policy, nn.Module):
    def __init__(self, embed_size):
        super(Policy, self).__init__()
        self.embed_size = embed_size
        #Similarity matrix
        self.W = nn.Parameter(torch.Tensor(self.embed_size, self.embed_size), requires_grad = True)

        #Similarity threshold
        self.threshold = nn.Parameter(torch.Tensor(1), requires_grad = True)

    
    def forward(self,
                clue_vector: torch.Tensor,
                options_matrix: torch.Tensor) -> torch.Tensor:

        return (torch.matmul(torch.matmul(clue_vector, self.W), torch.t(options_matrix))) - self.threshold


#Main method included only for testing purposes
def main():
    clue_vect = torch.empty(1,50)
    options_vect = torch.empty(10,50)
    policy = SimilarityThresholdPolicy(50)
    policy.forward(clue_vect, options_vect)

if __name__ == "__main__":
    main()