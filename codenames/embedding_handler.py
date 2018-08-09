from typing import List

import pickle

from scipy.spatial.distance import cosine
import numpy


class EmbeddingHandler:
    """
    Parameters
    ----------
    embedding_file : `str`
        Location of pickled embeddings
    """
    def __init__(self, embedding_file: str) -> None:
        # str -> numpy array
        self.embedding = pickle.load(open(embedding_file, "rb"))

    def get_word_vector(self, word: str) -> numpy.ndarray:
        word = word.lower()
        if word not in self.embedding:
            return None
        return self.embedding[word]

    def sort_options_by_similarity(self,
                                   clue: str,
                                   options: List[str],
                                   max_num_outputs: int = -1) -> List[str]:
        """
        Takes a clue and returns `max_num_outputs` number of `options` sorted by similarity with
        `clue`. If `max_num_outputs` is -1, returns all the options sorted by similarity.
        """
        word = clue.lower()
        if word not in self.embedding:
            return []
        word_vector = self.embedding[word]
        option_vectors = []
        for option in options:
            if option in self.embedding:
                option_vectors.append(self.embedding[option])
        distances = [cosine(word_vector, option_vector) for option_vector in option_vectors]
        sorted_options = [x[1] for x in sorted(zip(distances, options))]
        if max_num_outputs == -1:
            return sorted_options
        return sorted_options[:max_num_outputs]
