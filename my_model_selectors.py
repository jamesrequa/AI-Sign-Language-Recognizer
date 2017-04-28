import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """

        warnings.filterwarnings("ignore", message="divide by zero encountered in log")

        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")

        # TODO implement model selection based on BIC scores
        best_n_state = self.n_constant
        best_bic = None

        for n_state in range(self.min_n_components, self.max_n_components+1):
            # of features = d
            # of HMM states = n
            # of free parameters p = n*(n-1) + (n-1) + 2*d*n = n^2 + 2*d*n - 1

            try:
                hmm_model = self.base_model(n_state) #got this from base_model function
                d = len(self.X[0])
                p = n_state**2 + (2 * d * n_state) - 1
                logL = hmm_model.score(self.X, self.lengths)
                bic = -2 * logL + p * np.log(n_state)
                if best_bic is None or best_bic > bic:
                    best_bic = bic
                    best_n_state = n_state
            except:

                bic = float('-inf')

        return self.base_model(best_n_state)



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''



    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")

        best_dic = None
        best_n_state = self.n_constant

        # Create a list of all words except the current word
        other_words = list(self.words)
        other_words.remove(self.this_word)

        for n_state in range(self.min_n_components, self.max_n_components+1):
            try:
                # Fits model for this word
                hmm_model = self.base_model(n_state)
                # Gets score for this word
                logL_thisword = hmm_model.score(self.X, self.lengths)

                sum_other_scores = 0.0

                # Check the scores of all other words so we can compare
                for word in other_words:
                    # X, lengths corresponding to that other word
                    X, lengths = self.hwords[word]

                    # Total up the scores for all the other words
                    sum_other_scores += hmm_model.score(X, lengths)

                    # calculate the total number of other words, need to subtract one since this_word not included in average
                    m = len(self.words) - 1

                # Calculate DIC Score as DIC = log(P(thisword)) - average(log(P(otherwords)))
                dic =  logL_thisword - (sum_other_scores / m)

                # Keep the highest DIC score
                if best_dic is None or best_dic < dic:
                    best_dic = dic
                    best_n_state = n_state
            except:
                    pass

        return self.base_model(best_n_state)
