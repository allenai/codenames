import random
import numpy as np
from scipy.spatial.distance import cosine
from embedding_handler import EmbeddingHandler
import pickle

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

        with open("../data/" + self.dataset_file) as f:
            for line in f.readlines():
                self.data.append(line.lower().replace("\n", ""))

        # These params are used for the shuffle without replacement case
        self.dataset_loc = 0
        self.shuffled_data = self.data.copy()
        random.shuffle(self.shuffled_data)
        self.dataset_size = len(self.data)

        self.glove = None


    '''
    Most basic random sampling from a given dataset.  Assumes random samples WITH REPLACEMENT
    in between consecutive games. Optional to pass in number of assassins, positive, and negative words.
    By default, the number of neutral cards is determined by size-num_assassin-num_pos-num_neg.
    
    Note that the return is of the format:
        ASSASSIN;TEAM1;TEAM2;NEUTRAL
    where each group consists of comma-separated words from the word list.
        
    '''
    def sample_random(self, size=25, num_assassin=1, num_pos=9, num_neg=8,
                      guesser: EmbeddingHandler = None,
                      clue_giver: EmbeddingHandler = None):

        #This is the dataset restricted to valid words as per embeddings
        restricted_dataset = self.data.copy()
        if guesser != None:
            restricted_dataset = [w for w in restricted_dataset if w in guesser.embedding]

        if clue_giver != None:
            restricted_dataset = [w for w in restricted_dataset if w in clue_giver.embedding]


        board_words_str = ""
        words = random.sample(restricted_dataset, size)

        #Segments sampled words into gameboard's accepted format
        board_words_str += ';'.join([','.join(words[0:num_assassin]), ','.join(words[num_assassin:num_assassin + num_pos]),
                                     ','.join(words[num_assassin + num_pos:num_assassin + num_pos + num_neg]),
                                     ','.join(words[num_assassin + num_pos + num_neg:])])

        return board_words_str


    '''
       Random sampling for a dataset that assumes random samples WITHOUT REPLACEMENT in between consecutive games. 
       Optional to pass in number of assassins, positive, and negative words.
       By default, the number of neutral cards is determined by size-num_assassin-num_pos-num_neg.

       Note that the return is of the format:
           ASSASSIN;TEAM1;TEAM2;NEUTRAL
       where each group consists of comma-separated words from the word list.

       '''

    def sample_random_without_replacement(self, size=25, num_assassin=1, num_pos=9, num_neg=8,
                                          guesser: EmbeddingHandler = None,
                                          clue_giver: EmbeddingHandler = None):

        blacklisted_dataset = []
        if guesser != None:
            blacklisted_dataset.extend([w for w in self.data if w not in guesser.embedding])

        if clue_giver != None:
            blacklisted_dataset.extend([w for w in self.data if w not in clue_giver.embedding])

        board_words_str = ""

        #Edge case for if we're at the end of our shuffled list
        if self.dataset_loc + size > self.dataset_size:
            words = []
            words.extend([w for w in self.shuffled_data[self.dataset_loc:] if w not in blacklisted_dataset])
            random.shuffle(self.shuffled_data)
            self.dataset_loc = 0
            while len(words) < size:
                word = self.shuffled_data[self.dataset_loc]
                if word not in blacklisted_dataset:
                    words.append(word)
                self.dataset_loc += 1

        #Normal case, just adds the next words from the shuffled dataset
        else:
            words = []
            while len(words) < size:
                word = self.shuffled_data[self.dataset_loc]
                if word not in blacklisted_dataset:
                    words.append(word)
                self.dataset_loc += 1

        # Segments sampled words into gameboard's accepted format
        board_words_str += ';'.join(
            [','.join(words[0:num_assassin]), ','.join(words[num_assassin:num_assassin + num_pos]),
             ','.join(words[num_assassin + num_pos:num_assassin + num_pos + num_neg]),
             ','.join(words[num_assassin + num_pos + num_neg:])])

        return board_words_str


    #TODO finish this method.  Work in-progress, so commented out.
    '''
    This samples a "challenge" dataset by returning a dataset containing clusters of similar words
    (defined by GloVE cos sim).  
    
    In the default case: It selects a random word in the dataset as a "seed" word and 
    adds similar words (with cos distance less than Epsilon).  If not enough words are chosen, another 
    seed word is added and the process continues until "size" words have been chosen.
    
    If num_clusters is specified, there will be num_clusters seed words selected and words will be
    randomly added from each cluster until size words are found.
        
    Note that the return is of the format:
           ASSASSIN;TEAM1;TEAM2;NEUTRAL
    where each group consists of comma-separated words from the word list.
    '''
    '''def sample_similar_embeddings(self, size=25, num_assassin=1, num_pos=9, num_neg=8, num_clusters=-1, epsilon = 0.1):
        #Only need to intitialize once, but don't want to waste time upon object creation
        if(self.glove == None):
            f = open("./embeds/glove.6B.50d.txt", 'r')
            self.glove = {}
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                self.glove[word] = embedding

        if num_clusters != -1:
            #TODO the specified case here
            print("not implemented yet")
        else:
            words = []
            while len(words) < size:
                seed_word = ""
                while seed_word in words or seed_word == "":
                    seed_word = random.sample(self.data, 1)[0]
                import pdb; pdb.set_trace()


                seed_embed = np.zeros([50])
                for subword in seed_word.split(" "):
                    seed_embed +=
                for word in self.data:
                    if word == seed_word:
                        continue

                    if cosine(self.glove[seed_word], self.glove[word]) < epsilon:
                        words.append(word)
            random.shuffle(words)
        return words'''



def main():
    guesser_embed = EmbeddingHandler("./test_embeds.p")
    print("TESTS")
    d =  Dataset(dataset="CODENAMES")
    for i in range(100):
        sample = d.sample_random_without_replacement(guesser=guesser_embed, clue_giver=guesser_embed)
        print(sample)


if __name__ == "__main__":
    main()