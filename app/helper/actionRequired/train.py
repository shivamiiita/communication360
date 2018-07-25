# Adapted from http://www.nltk.org/book/ch06.html

from collections import Counter
import nltk
from nltk.util import ngrams
import random

def features(sentence):
    tokens = sentence.split()
    result =  Counter(tokens) \
            + Counter(nltk.ngrams(tokens, 2, pad_left=True, pad_right=True))

    result["SENTENCE_LENGTH"] = len(tokens)
    return result

