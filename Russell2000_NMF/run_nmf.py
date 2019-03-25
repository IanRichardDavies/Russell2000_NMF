# Import required packages
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

def iter_documents(top_directory):
	"""Iterate over all documents within the directory, yielding a dictionary
	The keys of the output dictionary are company names and the values is a preprocessing bags of words
	"""
	doc_dict = {}
	for root, dirs, files in os.walk(top_directory):
		for file in files:
			# read the entre document as one big string
			document = open(os.path.join(root, file)).read()
			document = preprocess_string(document)
			doc_dict[f'{file}'] = document
	return doc_dict

def produce_reference_corpus(dictionary):
	'''
	Dictionary should be the output from iter_documents
	The reference corpus is a list of lists
	reference_corpus[i][0] is a company name
	reference_corpus[i][0] is a bag of words
	Used later on when we need to match a company's name with its cluster
	'''
	reference_corpus = [[key, values] for key, values in dictionary.items()]
	return reference_corpus

def produce_corpus_list(dictionary):
	'''
	Dictioanry should be the ouput from iter_documents
	The corpus_list is a list of lists
	corpus_list[i] is a list of words
	'''
	corpus_list = [values for key, values in dictionary.items()]
	return corpus_list

def stem_function(corpus_list): 
	'''
	Uses nltk stemmer to stem words in corpus_list
	corpus_list is a list of lists (nested lists are lists of words)
	'''
	stemmer = nltk.stem.PorterStemmer()
	stemmed_list = []
	for group in corpus_list:
		nested_list = []
		for word in group:
			nested_list.append(stemmer.stem(word))
		stemmed_list.append(nested_list)
	return stemmed_list

def make_bigrams(prestemmed_list, stemmed_list):
	# Build the bigram models
	bigram = gensim.models.Phrases(prestemmed_list, min_count=5, threshold=10) 
	# This may seem redundant but will speed up computation
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	return [bigram_mod[doc] for doc in stemmed_list]

def make_trigrams(prestemmed_list, bigrammed_list):
	'''
	First argument should be the same unstemmed list that was passed into make_bigrams function
	Second argument should be the output list from the make_bigrams function
	'''
	bigram = gensim.models.Phrases(prestemmed_list, min_count=5, threshold=10) 
	trigram = gensim.models.Phrases(bigram[prestemmed_list], min_count=5, threshold=10) 
	# This may seem redundant but will speed up computation
	trigram_mod = gensim.models.phrases.Phraser(trigram)
	return [trigram_mod[doc] for doc in bigrammed_list]

def nmf_model(corpus_trigrams, num_topics):
	processed_corpus_str =  [' '.join(word) for word in corpus_trigrams]
	tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, ngram_range = (1,3), stop_words='english')
	tfidf = tfidf_vectorizer.fit_transform(processed_corpus_str)
	tfidf_feature_names = tfidf_vectorizer.get_feature_names()
	nmf = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
	nmf_W = nmf.transform(tfidf)
	nmf_H = nmf.components_
	return (nmf_H, nmf_W, tfidf_feature_names)

def display_topics(results, num_top_words, num_top_documents):
	'''
	results[0] = reference_corpus
	results[1] = corpus_list
	results[2] = nmf_H
	results[3] = nmf_W
	results[4] = tfidf_feature_names
	results[5] = classification_df	
	'''
	for topic_idx, token in enumerate(results[2]):
		print (f'Industry {topic_idx}:')
		print()
		print (", ".join([results[4][i] for i in token.argsort()[:-num_top_words - 1:-1]]))
		print()
		top_doc_indices = np.argsort(results[3][:,topic_idx] )[::-1][0:num_top_documents]
		for doc_index in top_doc_indices:
			print (results[0][doc_index][0][:-5])	
		print()

def make_classification_df(nmf_W):
	labels = []
	for i in range(len(nmf_W)):
		result = np.where(nmf_W[i] == np.max(nmf_W[i]))
		labels.append(int(result[0][0]))
	classification_df = pd.DataFrame(labels, columns = ['Industry'])
	return classification_df

def nmf_pipeline_function(year, num_topics, num_top_words = 40, num_top_documents = 10):
	'''
	Pipeline function that runs the full NMF process
	Returns a tuple.  If such tuple is called result:
	result[0] = reference_corpus
	result[1] = corpus_list
	result[2] = nmf_H
	result[3] = nmf_W
	result[4] = tfidf_feature_names
	result[5] = classification_df
	'''
	directory_path = f'{year}'
	init_corpus = iter_documents(directory_path)
	print("Created initial dictionary")
	reference_corpus = produce_reference_corpus(init_corpus)
	print('Created reference corpus')
	corpus_list = produce_corpus_list(init_corpus)  
	print('Created corpus list')
	stemmed_list = stem_function(corpus_list)
	print('Stemmed words')
	bigrammed_list = make_bigrams(corpus_list, stemmed_list)
	print('Created bigrams')
	corpus_trigrams = make_trigrams(corpus_list, bigrammed_list)
	print('Created trigrams')
	tup = nmf_model(corpus_trigrams, num_topics)
	print('Created NMF model')
	classification_df = make_classification_df(tup[1])
	return (reference_corpus, corpus_list, tup[0], tup[1], tup[2], classification_df)

if (__name__ == '__main__'):
	import sys
	nmf_pipeline_function((sys.argv[1]),(int(sys.argv[2])), (int(sys.argv[3])), (int(sys.argv[4])))