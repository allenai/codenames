import argparse
import os

from codenames.utils.file_utils import read_lines
import spacy

def main(args):
    corpus_location = args.corpus_location
    save_dir = args.save_dir

    tokenized_lowercased_file = os.path.join(os.path.basename(corpus_location) +
                                             "-tokenized_lc.txt")

    if not os.path.exists(tokenized_lowercased_file):
        nlp = spacy.load("en")
        with open(tokenized_lowercased_file) as tokenized_lc_file:
            for line in read_lines(corpus_location):
                sentence = line.split("\t")[1]
                tokens = [w.text.lower() for w in nlp(sentence)]
                tokenized_lc_file.write(' '.join(tokens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process results from the final AMTI task')
    parser.add_argument('--corpus_location',
                        type=str,
                        help='location of corpus')
    parser.add_argument('--save_dir',
                        type=str,
                        help='location of training data')

    args = parser.parse_args()
    main(args)