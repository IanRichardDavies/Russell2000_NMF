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

def sort_into_industries(classification_df, num_topics):
	'''
	df should be the output from nmf_pipeline_function() (tuple[5])
	num_topics should be the same entered as an argument in nmf_pipeline_function()
	This function returns a dictionary whose keys are industry labels and values are 
	company identification numbers
	'''
	companies_in_industry = {}
	for i in range(num_topics):
		topic_list = []
		for j in range(len(classification_df)):
			if int(classification_df.loc[j,'Industry']) == i:
				topic_list.append(j)
		companies_in_industry[str(i)] = topic_list
	return companies_in_industry

def get_sentiment(dictionary, corpus_list):
	'''
	Arg: dictionary should be the output from sort_into_industries()
	Arg: corpus_list should be the ouput from produce_corpus_list()
	This function returns a dictionary whose keys are industry labels and values
	are tuples such that tuple[0] is a company identification number and tuple[i] is its sentiment score
	'''
	from textblob import TextBlob
	corpus_strings =  [' '.join(word) for word in corpus_list]
	complete_sentiment = {}
	for industry, companies in dictionary.items():
		# companies_in_industry is a dictionary whose key is industry ID and values are lists of company IDs
		industry_sentiment = []
		for company_idx in companies:
			# A company_idx represents the id of each company
			text = TextBlob(corpus_strings[company_idx])
			sentiment = text.sentiment
			industry_sentiment.append((company_idx, sentiment[0]))
		complete_sentiment[industry] = industry_sentiment
	return complete_sentiment

def industry_average_sentiment(dictionary):
	'''
	Arg: dictionary should be the outut from get_sentiment()
	This function returns a dictionary whose keys are industry values 
	and whose values are the average sentiment score for each industry
	'''
	industry_sentiment = {}
	for industry, companies in dictionary.items():
		total = 0
		for company in companies:
			total += company[1]
		average = total / len(companies)
		industry_sentiment[industry] = average
	return industry_sentiment

def make_industry_sentiment_df(dictionary, nmf_H, tfidf_feature_names):
	'''
	arg: dictionary should be the output from industry_average_sentiment()
	arg: nmf_H should be the third item in the tuple returnred from nmf_pipeline_function()
		ex. if tup = nmf_pipeline_function(), the nmf_H = tup[2]
	arg: tfidf_feature_names should be the the fifth item in the tuple returned from nmf_pipeline_function()
		ex. if tup = nmf_pipeline_function(), then tfidf_feature_names = tup[4]
	This function returns a dataframe with sentiment analysis by industry
	'''
	industry_keywords = []
	industry_list = []
	sentiment_list = []
	for topic_idx, token in enumerate(nmf_H):
		# nmf_H is the matrix whose rows are topics and columns are word tokens
		# token is a vector whose length is the size of the corpus
		keywords = [tfidf_feature_names[i] for i in token.argsort()[:-40-1:-1]]
		industry_keywords.append(keywords)
	for industry, sentiment in dictionary.items():
		industry_list.append(industry)
		sentiment_list.append(round(sentiment,4))
	industry_sentiment_df = pd.DataFrame({'Industry': industry_list, 
										  'Sentiment': sentiment_list, 
										  'Industry_Keywords': industry_keywords})
	pd.set_option('display.max_colwidth', -1)
	return industry_sentiment_df

def make_company_sentiment_df(dictionary, nmf_H, tfidf_feature_names, reference_corpus):
	'''
	arg: dictionary should be the output from industry_average_sentiment()
	arg: nmf_H should be the third item in the tuple returnred from nmf_pipeline_function()
		ex. if tup = nmf_pipeline_function(), the nmf_H = tup[2]
	arg: tfidf_feature_names should be the the fifth item in the tuple returned from nmf_pipeline_function()
		ex. if tup = nmf_pipeline_function(), then tfidf_feature_names = tup[4]
	arg: reference_corpus should be the output from produce_reference_corpus()
	This function returns a dataframe showing the most positive and most negative companies in each industry
	'''
	industry_keywords = []
	industry_list = []
	most_positive = []
	most_negative = []
	for topic_idx, token in enumerate(nmf_H):
		# nmf_H is the matrix whose rows are topics and columns are word tokens
		# token is a vector whose length is the size of the corpus
		keywords = [tfidf_feature_names[i] for i in token.argsort()[:-40-1:-1]]
		industry_keywords.append(keywords)
		industry_list.append(topic_idx)
	for industry, companies in dictionary.items():
		companies = sorted(companies, key = lambda tup: -tup[1])
		most_pos = [int(x[0]) for x in companies[:10]]
		most_neg = [int(x[0]) for x in companies[:-10:-1]]
		pos_list = [reference_corpus[x][0][:-5].capitalize() for x in most_pos]              
		neg_list = [reference_corpus[x][0][:-5].capitalize() for x in most_neg]
		most_positive.append(pos_list)
		most_negative.append(neg_list)

	company_sentiment_df = pd.DataFrame({'Industry': industry_list, 
										  'Most_Postive': most_positive, 
										  'Most_Negative': most_negative,
										  'Industry_Keywords': industry_keywords})
	pd.set_option('display.max_colwidth', -1)
	return company_sentiment_df

def industry(year, num_topics, results):
	'''
	arg: year: desired year of analysis
	arg: num_tropics: number of topics/industries/clusters
	Returns a dataframe with the desired sentiment analysis
	reference_corpus = results[0]
	corpus_list = results[1]
	nmf_H = results[2]   
	nmf_W = results[3]
	tfidf_feature_names = results[4]
	classification_df = results[5]
	'''
	companies_in_industry = sort_into_industries(results[5], num_topics)
	complete_sentiment = get_sentiment(companies_in_industry, results[1])
	industry_sentiment = industry_average_sentiment(complete_sentiment)
	industry_sentiment_df = make_industry_sentiment_df(industry_sentiment, results[2], results[4])
	return industry_sentiment_df

def company(year, num_topics, results):
	'''
	arg: year: desired year of analysis
	arg: num_tropics: number of topics/industries/clusters
	Returns a dataframe with the desired sentiment analysis
	reference_corpus = results[0]
	corpus_list = results[1]
	nmf_H = results[2]   
	nmf_W = results[3]
	tfidf_feature_names = results[4]
	classification_df = results[5]
	'''
	companies_in_industry = sort_into_industries(results[5], num_topics)
	complete_sentiment = get_sentiment(companies_in_industry, results[1])
	industry_sentiment = industry_average_sentiment(complete_sentiment)
	company_sentiment_df = make_company_sentiment_df(complete_sentiment, results[2], results[4], results[0])
	return company_sentiment_df


if (__name__ == '__main__'):
	import sys
	industry((sys.argv[1]),(int(sys.argv[2])),(argv[3]))
	company((sys.argv[1]),(int(sys.argv[2])),(sys.argv[3]))