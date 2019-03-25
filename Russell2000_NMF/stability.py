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

def create_set_companies(list_of_years):
    # initialize set
    first_year = list_of_years[0]
    path = f'{first_year}'
    companies_set = {f for f in listdir(path) if isfile(join(path, f))}
    # iterate over the rest
    for i in range(1,len(list_of_years)):
        path = f'{list_of_years[i]}'
        companies_temp = {f for f in listdir(path) if isfile(join(path, f))}
        companies_set = companies_set.intersection(companies_temp)
    return companies_set

def iter_hist_documents(year, companies_set):
    doc_dict = {}
    for company in companies_set:
        for root, dirs, files in os.walk(f'{year}'):
            for file in files:
                if str(file) == str(company):
                    document = open(os.path.join(root, file)).read() # read the entire document, as one big string
                    document = preprocess_string(document)
                    doc_dict[f'{file}'] = document
    return doc_dict

def produce_reference_corpus(dictionary):
	'''
	Dictionary should be the output from iter_documents
	The reference corpus is a list of lists
	reference_corpus[i][0] is a company name
	reference_corpus[i][1] is a bag of words
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

def label_companies(nmf_W):
	labels = []
	for i in range(len(nmf_W)):
		result = np.where(nmf_W[i] == np.max(nmf_W[i]))
		labels.append(int(result[0][0]))
	return labels

def stability_pipeline_function(year, num_topics, companies_set):
	'''
	Pipeline function that runs the full NMF process
	'''
	init_corpus = iter_hist_documents(year, companies_set)
	reference_corpus = produce_reference_corpus(init_corpus)
	corpus_list = produce_corpus_list(init_corpus)  
	stemmed_list = stem_function(corpus_list)
	bigrammed_list = make_bigrams(corpus_list, stemmed_list)
	corpus_trigrams = make_trigrams(corpus_list, bigrammed_list)
	tup = nmf_model(corpus_trigrams, num_topics)
	labels = label_companies(tup[1])
	return labels

def stability(years_list, num_topics):
	'''
	Function that returns adjusted rand score for each sequential pairs of years in years_list
	arg: years_list: list of targeted years, should only include years entered into nmf() call
	arg: num_topics: should be the same argument as entered into nmf() call 
	arg: results: is the full results from nmf() call 
	len(results) should equal number of years
	results[i][0] = reference_corpus
	results[i][1] = corpus_list
	results[i][2] = nmf_H
	results[i][3] = nmf_W
	results[i][4] = tfidf_feature_names
	results[i][5] = classification_df
	'''
	companies_set = create_set_companies(years_list)
	from sklearn.metrics import adjusted_rand_score
	labels_list = []
	for year in years_list:
		labels = stability_pipeline_function(year, num_topics, companies_set)
		print(f'Completed {year}')
		labels_list.append(labels)
	scores = []
	for i in range(len(labels_list)-1):
		true = labels_list[i]
		pred = labels_list[i+1]
		score = adjusted_rand_score(true, pred)
		scores.append(score)
	return scores

if (__name__ == '__main__'):
	import sys
	stability((list(sys.argv[1])),(int(sys.argv[2])))
