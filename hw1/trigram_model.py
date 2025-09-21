import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2025 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 

    tuple size is n
    """
    if n < 1 or not sequence:
        return []
    #unigram case - do not include STOP marker
    if n > 1:
        padded_sequence = ['START'] * (n - 1) + sequence + ['STOP']
    #all the other ngram cases
    elif n == 1:
        padded_sequence = sequence 

    length = len(padded_sequence)
    res = []
    l, r = 0, n
    while r <= length:
        res.append(tuple(padded_sequence[l:r]))
        l += 1
        r += 1
     
    return res

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) 
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        #Generate 3 separate ngrams
        for sentence in corpus:
            unigram = get_ngrams(sentence, 1)
            bigram = get_ngrams(sentence, 2)
            trigram = get_ngrams(sentence, 3)
        
            for gram in unigram:
                self.unigramcounts[gram] += 1
            for gram in bigram:
                self.bigramcounts[gram] += 1
            for gram in trigram:
                self.trigramcounts[gram] += 1

        #store total_unigrams for future use
        self.total_unigrams = sum(self.unigramcounts.values())
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        u, v, w = trigram
        trigram_count = self.trigramcounts[(u, v, w)]
        bigram_count = self.bigramcounts[(u, v)]

        if bigram_count == 0: 
            return 1 / len(self.lexicon)
        return trigram_count / bigram_count

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        u, v = bigram
        bigram_count = self.bigramcounts[(u, v)]
        unigram_count = self.unigramcounts[(u,)]
        if unigram_count == 0:
            return 1 / len(self.lexicon)
        return bigram_count / unigram_count
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        unigram_count = self.unigramcounts[unigram]
        total = self.total_unigrams
        if total == 0:
            return 1 / len(self.lexicon)
        return unigram_count / total

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        u, v, w = trigram
        return (
            lambda1 * self.raw_trigram_probability((u, v, w)) + 
            lambda2 * self.raw_bigram_probability((v, w)) + 
            lambda3 * self.raw_unigram_probability((w,))
        )
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        all_trigrams = get_ngrams(sentence, 3)
        res = 0
        for trigram in all_trigrams:
            res += math.log2(self.smoothed_trigram_probability(trigram))
        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        return float("inf") 


# def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

#         model1 = TrigramModel(training_file1)
#         model2 = TrigramModel(training_file2)

#         total = 0
#         correct = 0       
 
#         for f in os.listdir(testdir1):
#             pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
#             pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
#             # .. 
    
#         for f in os.listdir(testdir2):
#             # .. 
        
#         return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

