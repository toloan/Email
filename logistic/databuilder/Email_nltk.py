import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from os import listdir
from os.path import isfile, join
from collections import Counter
from Email import Email

class Email_nltk(Email) :

	def remove_stopwords(self):
		words = re.sub(r'[.!,;?"#$%&\'()*+-/@<>:0-9{}\[\]/=~_`|\\]', ' ', self.content).split()
		# words = list(set(words) - set(stopwords.words('english')))
		words_without_stopwords = [word.lower() for word in words if word not in stopwords.words('english')]
		return words_without_stopwords

	def get_list(self):
		lmtzr = WordNetLemmatizer()
		words = self.remove_stopwords()
		lemmatized_words = [lmtzr.lemmatize(word, pos[0].lower()) if  pos[0].lower() in ['a','n','v'] else lmtzr.lemmatize(word) for word, pos in pos_tag(words)] 
		return lemmatized_words

	def get_matrix(self, dictionary):
		words = self.get_list()
		return  [words.count(element) for element in dictionary]


