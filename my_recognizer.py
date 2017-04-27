import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    test_sequences = list(test_set.get_all_Xlengths().values())

    for test_X, test_lengths in test_sequences:
        logL_w = dict()
        for word, model in models.items():
            try:
                logL_w[word] = model.score(test_X, test_lengths)

            except:
                logL_w[word] = float('-inf')

        probabilities.append(logL_w)

    for prob in probabilities:
        guess_word = max(prob, key = prob.get)
        guesses.append(guess_word)

    return probabilities, guesses
