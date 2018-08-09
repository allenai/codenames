import argparse
import os

from codenames.utils.file_utils import read_lines
import spacy
from gensim.models import Word2Vec
import logging

logging.getLogger().setLevel(logging.INFO)


def main(args):
    corpus_location = args.corpus_location
    save_dir = args.save_dir
    workers = args.workers

    output_weights_file = os.path.join(save_dir, "weights.txt")
    model_file = os.path.join(save_dir, "model.pkl")

    tokenized_lowercased_filename = os.path.join(os.path.dirname(corpus_location),
                                             os.path.basename(corpus_location) + "-tokenized_lc.txt")

    all_sentences = []
    if not os.path.exists(tokenized_lowercased_filename):
        print("Writing to file: " + tokenized_lowercased_filename)
        nlp = spacy.load("en")
        with open(tokenized_lowercased_filename, "w") as tokenized_lc_file:
            for line in read_lines(corpus_location):
                sentence = line.split("\t")[1]
                tokens = [w.text.lower() for w in nlp(sentence)]
                sentence = ' '.join(tokens)
                tokenized_lc_file.write(sentence)
                tokenized_lc_file.write("\n")
                all_sentences.append(sentence)
        tokenized_lc_file.close()
    else:
        all_sentences = read_lines(tokenized_lowercased_filename)

    logging.info("Found {} sentences".format(len(all_sentences)))

    model = Word2Vec(
        sentences=all_sentences,
        size=300,
        window=10,
        workers=workers,
        negative=10,
        min_count=50,
        sg=1,
        iter=10
    )

    model.wv.save_word2vec_format(output_weights_file)
    model.save(model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process results from the final AMTI task')
    parser.add_argument('--corpus_location',
                        type=str,
                        help='location of corpus')
    parser.add_argument('--save_dir',
                        type=str,
                        help='location of training data')
    parser.add_argument('--workers',
                        type=int,
                        help='number of workers')

    args = parser.parse_args()
    main(args)