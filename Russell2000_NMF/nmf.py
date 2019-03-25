import re
import numpy as np
import pandas as pd
from pprint import pprint
import os
import nltk
nltk.download('stopwords')
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from os import listdir
from os.path import isfile, join
import datetime
import requests
import csv
import Russell2000_NMF.run_nmf as run_nmf
import Russell2000_NMF.sentiment as sentiment
import Russell2000_NMF.stability as stability

def nmf(year, num_topics, num_top_words = 40, num_top_documents = 10):
	'''
	Function that runs a NMF clustering model for the desired years
	arg: year: target year
	arg: num_topics: number of industries/topics/clusters
	arg: num_top_words: number of keywords returned for each industry/cluster
	arg: num_top_documents: number of companies returned for each industry/cluster
	results will be a tuple
	results[0] = reference_corpus
	results[1] = corpus_list
	results[2] = NMF H matrix (topics x words)
	results[3] = NMF W matrix (documents x topics)
	results[4] = tfidf_feature_names
	results[5] = classification_df
	'''
	results = run_nmf.nmf_pipeline_function(year, num_topics, num_top_words, num_top_documents)
	print('Output tuple:')
	print('results[0] = labeled preprocessed bag of words for each document')
	print('results[1] = unlabeled preprocessed bag of words for each document')
	print('results[2] = NMF H matrix')
	print('results[3] = NMF W matrix')
	print('results[4] = feature names of TF-IDF matrix')
	print('results[5] = dataframe with clustering results')
	return results

def display(results, num_top_words = 40, num_top_documents = 10):
	'''
	Function that display results of the clustering
	arg: results: should a flat tuple of a single year's output from nmf() 
	'''
	run_nmf.display_topics(results, num_top_words, num_top_documents)

def industry_sentiment(year, num_topics, results):
	'''
	Function that return the overall sentiment for each industry
	arg: year: should only be a year included in the nmf() call
	arg: num_topics: should be the the same argument entered for nmf() call
	'''	
	result = sentiment.industry(year, num_topics, results)
	return result

def company_sentiment(year, num_topics, results):
	'''
	Function that return the overall sentiment for each industry
	arg: year: should only be a year included in the nmf() call
	arg: num_topics: should be the the same argument entered for nmf() call
	arg: results: should be a flat tuple of a single year's output from nmf()
	'''	
	result = sentiment.company(year, num_topics, results)
	return result

if (__name__ == '__main__'):
	import sys
	nmf((sys.argv[1]),(int(sys.argv[2])))
	