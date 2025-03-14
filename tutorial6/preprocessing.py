#import nltk
#nltk.download()
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import os

MAX_LINES = 100
DIR = 'hi'
def readFileIntoLists(filename):
    #lines = []
    full_filename = os.path.join(DIR, filename)
    try:
        with open(full_filename, encoding="utf8") as csvFile:
            positive = []
            negative = []
            reader = csv.reader(csvFile)
            data = list(reader)
            for row in data[1:]:
                if row[1] == '0':
                    positive.append(row[0])
                else:
                    negative.append(row[0])
            return positive, negative
    except:
        print('something went wrong with the file', full_filename)
        return [], []
def createLexicon(pos: list[str], neg: list[str], lemmatizer: WordNetLemmatizer):
    """
    This function should generate a lexicon (list of words) from the provided positive and negative samples.
    - Tokenize words using word_tokenize()
    - Lemmatize them using lemmatizer.lemmatize()
    - Use Counter() to count word occurrences and filter based on frequency criteria
    """
def sampleHandling(sample: list[str], lexicon, classification, lemmatizer: WordNetLemmatizer):
    """
    This function should convert a given sample into a feature set using the provided lexicon.
    - Tokenize and lemmatize words
    - Create a feature vector where each word in the lexicon is represented as a frequency count
    - Append classification labels to the feature vector
    """
    # Write your code here
    pass
def processData(filename):
    try:
        pos, neg = readFileIntoLists(filename)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError('No positive or negative samples')
        lemmatizer = WordNetLemmatizer()
        lexicon = createLexicon(pos, neg, lemmatizer)
        features = []
        features += sampleHandling(pos, lexicon, [1, 0], lemmatizer)
        features += sampleHandling(neg, lexicon, [0, 1], lemmatizer)
        random.shuffle(features)
        return features
    except ImportError:
        print('something went wrong')

#[[0,0,0,0,0,0,10,0,2], [0, 1], [], []]
if __name__ == '__main__':
    processData('Train.csv')