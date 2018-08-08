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

    '''
    Most basic random sampling from a given dataset.  Assumes random samples WITH REPLACEMENT
    in between consecutive games. Optional to pass in number of assassins, positive, and negative words.
    By default, the number of neutral cards is determined by size-num_assassin-num_pos-num_neg.
    
    guesser/clue_givers are used to determine blacklists.
    
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
       
       guesser/clue_givers are used to determine blacklists.

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


    '''
    This samples a "challenge" dataset by returning a dataset containing clusters of similar words
    (defined by GloVE cos sim). 
    
    similarity: EmbeddingHandler is the embedding space we will use to calculate similarity.
    
    guesser/clue_givers are used to determine blacklists.
    
    In the default case: It selects a random word in the dataset as a "seed" word and 
    adds similar words (with cos distance less than Epsilon).  If not enough words are chosen, another 
    seed word is added and the process continues until "size" words have been chosen.
    
    If num_clusters is specified, there will be num_clusters seed words selected and words will be
    randomly added from each cluster until size words are found.
        
    Note that the return is of the format:
           ASSASSIN;TEAM1;TEAM2;NEUTRAL
    where each group consists of comma-separated words from the word list.
    '''
    #TODO discuss epsilon value
    def sample_similar_embeddings(self, similarity: EmbeddingHandler, size=25, num_assassin=1, num_pos=9, num_neg=8, num_clusters=-1,
                                  guesser: EmbeddingHandler = None,
                                  clue_giver: EmbeddingHandler = None,
                                  epsilon = 0.3):

        #Return string
        board_words_str = ""

        blacklisted_dataset = []
        if guesser != None:
            blacklisted_dataset.extend([w for w in self.data if w not in guesser.embedding])

        if clue_giver != None:
            blacklisted_dataset.extend([w for w in self.data if w not in clue_giver.embedding])

        #Since we are using `similarity`'s metric, each potential word has to be in similarity
        blacklisted_dataset.extend([w for w in self.data if w not in similarity.embedding])

        if num_clusters != -1:
            words = []
            tries = 1000
            counter = 0
            while len(words) < size and counter < tries:
                words = []
                seed_words = []
                while(len(seed_words) < num_clusters):
                    seed_word = random.sample(self.data, 1)[0]
                    if seed_word not in seed_words and seed_word not in blacklisted_dataset:
                        seed_words.append(seed_word)
                words.extend([w for w in seed_words])

                similar_words = []

                for word in self.data:
                    if word in seed_words or word in blacklisted_dataset or word in similar_words:
                        continue
                    for seed_word in seed_words:
                        seed_embed = similarity.embedding[seed_word]
                        if cosine(seed_embed, similarity.embedding[word]) < epsilon:
                            similar_words.append(word)
                if len(similar_words) >= size-num_clusters:
                    words.extend(random.sample(similar_words, size-num_clusters))
                counter += 1
            if len(words) < size:
                raise ValueError("Cannot find enough words with the cluster/size combo.  Try increasing num_clusters or"
                                 "increasing epsilon")

            random.shuffle(words)
        else:
            words = []
            while len(words) < size:
                seed_word = ""
                while seed_word in words or seed_word == "" or seed_word in blacklisted_dataset:
                    seed_word = random.sample(self.data, 1)[0]
                words.append(seed_word)
                seed_embed = similarity.embedding[seed_word]
                for word in self.data:
                    if word == seed_word or word in blacklisted_dataset:
                        continue

                    if cosine(seed_embed, similarity.embedding[word]) < epsilon:
                        words.append(word)

            random.shuffle(words)
        # Segments sampled words into gameboard's accepted format
        board_words_str += ';'.join(
            [','.join(words[0:num_assassin]), ','.join(words[num_assassin:num_assassin + num_pos]),
             ','.join(words[num_assassin + num_pos:num_assassin + num_pos + num_neg]),
             ','.join(words[num_assassin + num_pos + num_neg:])])

        return board_words_str



def main():
    guesser_embed = EmbeddingHandler("./test_embeds.p")
    print("TESTS")
    d =  Dataset(dataset="CODENAMES")
    #for i in range(100):
    sample = d.sample_similar_embeddings(similarity=guesser_embed, guesser=guesser_embed, clue_giver=guesser_embed, num_clusters=5)
    print(sample)


if __name__ == "__main__":
    main()