from gensim.models import Word2Vec

from codenames.utils import file_utils
from codenames.utils.file_utils import read_lines
from random import choices
import random

words = [w.replace(" ", "_") for w in read_lines("codenames/gameplay/words.txt")]

all_sentences = []
for i in range(10000):
    num_words = random.randint(4, 10)
    all_sentences.append(
        choices(words, k=num_words)
    )

model = Word2Vec(
    sentences=all_sentences,
    size=300,
    window=10,
    workers=1,
    negative=10,
    min_count=50,
    sg=1,
    iter=10
)

model.wv.save_word2vec_format("tests/fixtures/model.txt")
model.save("tests/fixtures/model.dat")