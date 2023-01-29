import re
import string
import json
import nltk
import warnings

warnings.filterwarnings("ignore")
nltk.download('popular', quiet=True)
stopword = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

# Load the loglikelihood file
with open('artifacts/loglikelihood.json', 'rb') as fp:
    loglikelihood = json.load(fp)

def clean_review(review):
    """
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review.

    """
    # Convert lowercase
    review_cleaned = review.lower()

    # Remove links
    review_cleaned = re.sub(r"http\S+", "", review_cleaned)  # replaces URLs starting with http
    review_cleaned = re.sub(r"www.\S+", "", review_cleaned)  # replaces URLs starting with www
    review_cleaned = re.sub(r"\S+.com$", "", review_cleaned)  # replaces URLs ending with .com

    # Remove punctuation
    review_cleaned = "".join([char for char in review_cleaned if
                              char not in string.punctuation])  # Causes spacing issues like "horrendousbr" from "horrendous.<br />"
    # Remove stopwords
    review_cleaned = " ".join([word for word in re.split('\W+', review_cleaned) if word not in stopword])

    # Stem the words
    review_cleaned = " ".join([ps.stem(word) for word in re.split('\W+', review_cleaned)])

    return review_cleaned


def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''

    # process the review to get a list of words
    word_l = clean_review(review).split(" ")

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]
            print("        Word: ", word, "-> ", loglikelihood[word])

    if total_prob < 0:
        return 0
    else:
        return 1


def text_classifier():
    """
    Loads the NB model and allows users to classify new reviews

    Params:
        review: a users input
        loglikelihood: a dictionary of words mapping to numbers, loaded automatically
    Return:
        classification: probabilities of each word, and final classification

    """
    print("""
####################################################
   _____            _   _                      _   
  / ____|          | | (_)                    | |  
 | (___   ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_ 
  \___ \ / _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __|
  ____) |  __/ | | | |_| | | | | | |  __/ | | | |_ 
 |_____/ \___|_| |_|\__|_|_|_|_| |_|\___|_| |_|\__|
  / ____| |             (_)/ _(_)                  
 | |    | | __ _ ___ ___ _| |_ _  ___ _ __         
 | |    | |/ _` / __/ __| |  _| |/ _ \ '__|        
 | |____| | (_| \__ \__ \ | | | |  __/ |           
  \_____|_|\__,_|___/___/_|_| |_|\___|_|           
                                                   
####################################################
                                                   """)
    print("Welcome to the movie review classifier!\n")
    print("Instructions:")
    print("Enter your review and we will detect the sentiment.")
    print("Example: 'This is a great movie', or 'this is a terrible movie'\n")
    print("To exit the application, enter 'X' \n")

    while True:
        user_input = input("Review: ")

        if str(user_input) == "x" or str(user_input) == "X":
            print("Exiting now!")
            break

        else:
            nb_predict = naive_bayes_predict(user_input, 0.0, loglikelihood)
            if nb_predict == 1:
                print("        Negative Sentiment Detected (1)")
            elif nb_predict == 0:
                print("        Positive Sentiment Detected (0)")
            else:
                print("Unknown error. Please contact admin")


if __name__ == '__main__':
    text_classifier()

