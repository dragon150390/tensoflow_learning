import nltk
# The first chanllenge when I will come up against after I learne based on the demos
# preprogress the data We need to faced with that chanllenge
# simple preprocessing the languge Data

#nltk.download()
from nltk.tokenize import word_tokenize
import numpy as np
import random 
import pickle 
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 10000
def create_lexicon ():
	lexicon = []
	
	with open(pos,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			# call word_tokenize from nltk
			all_words = word_tokenize(l)
			lexicon += list(all_words);
			
	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)
	
	
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	# count the word occuranece 
	# the count of words less than 1000 and greater than 50
	for w in w_counts:
		if 1000> w_counts[w] >50:
			l2.append(w)
	print (len(l2))
	return l2

