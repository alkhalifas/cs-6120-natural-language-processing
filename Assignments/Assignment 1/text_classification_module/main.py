import numpy as np
import math
import pandas as pd
import re
import string
import random

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.metrics import confusion_matrix


from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
nltk.download('popular', quiet=True)

stopword = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

def text_classifier():
    print("Welcome to the movie classifier!")


if __name__ == '__main__':
    text_classifier()

