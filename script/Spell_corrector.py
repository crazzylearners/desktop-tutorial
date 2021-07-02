import re
import string
from collections import Counter
import numpy as np

def extract_vocabs_(text):
    words = []
    words += re.findall(r'\w+', text.lower())
    vocabs = set(words)
    word_counts = Counter(words)
    total_words = float(sum(word_counts.values()))
    word_probas = {word: word_counts[word] / total_words for word in vocabs}
    return (vocabs,word_probas)

def _level_one_edits(word):
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l,r in splits if r]
    swaps = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r)>1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in letters]
    inserts = [l + c + r for l, r in splits for c in letters] 
    return set(deletes + swaps + replaces + inserts)

def _level_two_edits(word):
    return set(e2 for e1 in _level_one_edits(word) for e2 in _level_one_edits(e1))

def check_(word, text):
    vocabs = text[0]
    word_probas = text[1]
    candidates = _level_one_edits(word) or _level_two_edits(word) or [word]
    valid_candidates = [w for w in candidates if w in vocabs]
    return sorted([(c, word_probas[c]) for c in valid_candidates], key=lambda tup: tup[1], reverse=True)