'''Objects and methods to support text corpus storage and manipulation'''
import numpy as np
import pandas as pd
import re
import string

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer

# Kludge to prevent isalph() from checking for non-ASCII characters
def is_alpha(char):
    return char in string.ascii_lowercase

def keep_alphanumeric(doc):
    doc = [char for char in doc]
    out = ''
    for char in doc:
        good = is_alpha(char) or char.isnumeric() or char in [' ', '.']
        if good:
            out += char
        else:
            out += ' '
    return out

def remove_special(docs):
    docs = [doc.lower() for doc in docs]
    docs = [keep_alphanumeric(doc) for doc in docs]
    return docs

# Converts numerals to text, i.e. 1 to 'one'
def remove_numerals(docs):
    docs = [doc.replace('0', ' zero ') for doc in docs]
    docs = [doc.replace('1', ' one ') for doc in docs]
    docs = [doc.replace('2', ' two ') for doc in docs]
    docs = [doc.replace('3', ' three ') for doc in docs]
    docs = [doc.replace('4', ' four ') for doc in docs]
    docs = [doc.replace('5', ' five ') for doc in docs]
    docs = [doc.replace('6', ' six ') for doc in docs]
    docs = [doc.replace('7', ' seven ') for doc in docs]
    docs = [doc.replace('8', ' eight ') for doc in docs]
    docs = [doc.replace('9', ' nine ') for doc in docs]
    return docs

# Removes extra whitespace for a list of text strings. Kludgy, but useful.
def remove_whitespace(docs):
	docs = [' '.join(doc.split()) for doc in docs]
	docs = [doc.rstrip() for doc in docs]
	docs = [doc.lstrip() for doc in docs]
	return docs

# Removes 'nan' strings for a list of strings
def remove_nan(docs):
    docs = [doc.replace('nan', '') for doc in docs]
    return docs

# Function that cleans up a free-text column
def clean_text(docs):
    docs = [doc.lower() for doc in docs]
    docs = remove_special(docs)
    docs = [doc.replace('yom', ' year old male ') for doc in docs]
    docs = [doc.replace('yof', ' year old female ') for doc in docs]
    docs = [doc.replace('ym', ' year old male ') for doc in docs]
    docs = [doc.replace('yf', ' year old female ') for doc in docs]
    docs = [doc.replace('yowm', ' year old male ') for doc in docs]
    docs = [doc.replace('yowf', ' year old female ') for doc in docs]
    docs = [doc.replace('yo m', ' year old male ') for doc in docs]
    docs = [doc.replace('y o m', ' year old male ') for doc in docs]
    docs = [doc.replace('yo f', ' year old female ') for doc in docs]
    docs = [doc.replace('y o f', ' year old female ') for doc in docs]
    docs = [doc.replace(' yo ', ' year old ') for doc in docs]
    docs = [doc.replace('dx', ' diagnosis ') for doc in docs]
    docs = [doc.replace(' d x ', ' diagnosis ') for doc in docs]
    docs = [doc.replace(' c o ', ' complains of ') for doc in docs]
    docs = [doc.replace('bibems', ' brought in by ems ') for doc in docs]
    docs = [doc.replace(' pt ', ' patient ') for doc in docs]
    docs = [doc.replace(' pts ', ' patients ') for doc in docs]
    docs = [doc.replace(' lac ', ' laceration ') for doc in docs]
    docs = [doc.replace(' lt ', ' left ') for doc in docs]
    docs = [doc.replace(' rt ', ' right ') for doc in docs]
    docs = [doc.replace(' sus ', ' sustained ') for doc in docs]
    docs = [doc.replace('fx', ' fracture ') for doc in docs]
    docs = [doc.replace('bldg', ' building ') for doc in docs]
    docs = [doc.replace(' s p ', ' status post ') for doc in docs]
    docs = [doc.replace(' w ', ' with ') for doc in docs]
    docs = [doc.replace(' gsw ', ' gun shot wound ') for doc in docs]
    docs = [doc.replace(' etoh ', ' ethanol ') for doc in docs]
    docs = [doc.replace(' loc ', ' loss of consciousness ') for doc in docs]
    docs = [doc.replace('pta', ' prior to arrival ') for doc in docs]
    docs = [doc.replace(' x ', ' for ') for doc in docs]
    docs = [doc.replace(' chi ', ' closed head injury ') for doc in docs]
    docs = [doc.replace(' 2 2 ', ' secondary to ') for doc in docs]
    docs = [doc.replace('lbp', ' low blood pressure ') for doc in docs]
    docs = [doc.replace(' htn ', ' hypertension ') for doc in docs]
    docs = [doc.replace(' pw ', ' puncture wound ') for doc in docs]
    docs = remove_whitespace(docs)
    return docs
