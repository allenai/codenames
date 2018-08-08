import random

dataset_codes = {"CODENAMES": "codenames_words.txt", "SCIENCE": "science_words.txt",
                 "COMMON": "common_nouns_extrinsic.txt", "PROPER": "proper_nouns_extrinsic.txt"}

class Dataset():
    '''
    Initializes dataset from text file.  Assumes dataset is a code found in dataset_codes
    and that the text file has one word per line
    '''
    def __init__(self, dataset="CODENAMES"):
        if dataset not in dataset_codes:
            raise ValueError('Expected dataset to be one of ' + str(dataset_codes.keys()))
        self.dataset = dataset
        self.dataset_file = dataset_codes[dataset]
        self.data = []
        #TODO re-do this code for relative structure once integrated
        with open("./data/" + self.dataset_file) as f:
            for line in f.readlines():
                self.data.append(line.lower().replace("\n", ""))

    '''
    Most basic random sampling from a given dataset.  Assumes random samples WITH REPLACEMENT
    in between consecutive games. Optional to pass in number of assassins, positive, and negative words.
    By default, the number of neutral cards is determined by size-num_assassin-num_pos-num_neg.
    
    Note that the return is of the format:
        ASSASSIN;TEAM1;TEAM2;NEUTRAL
    where each group consists of comma-separated words from the word list.
        
    '''
    def sample_random(self, size=25, num_assassin=1, num_pos=9, num_neg=8):
        num_neutral = size - num_assassin - num_pos - num_neg
        board_words_str = ""
        words = random.sample(self.data, size)

        #Segments sampled words into gameboard's accepted format
        board_words_str += ';'.join([','.join(words[0:num_assassin]), ','.join(words[num_assassin:num_assassin + num_pos]),
                                     ','.join(words[num_assassin + num_pos:num_assassin + num_pos + num_neg]),
                                     ','.join(words[num_assassin + num_pos + num_neg:])])
        return board_words_str


def main():
    print("TESTS")
    d =  Dataset(dataset="CODENAMES")
    sample = d.sample_random()
    print(sample)


if __name__ == "__main__":
    main()