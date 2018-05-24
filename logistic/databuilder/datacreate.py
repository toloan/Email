import os
import Email
import Email_nltk
from collections import Counter
def dictionary(path):
	
	dictionary=list()
	for filename in os.listdir(path):
		try: 
			file=open(path+"//"+filename)
			email=Email.Email(file)
			file.close()
			dictionary=list(set(dictionary+email.get_list()))
		except IOError:
			print("Error: file: "+file_name + " does not exist")
	dictionary.sort()
	return dictionary	

def dictionary_nltk(path):
	
	dictionary=list()
	for filename in os.listdir(path):
		if "spm" in filename:
			try: 
				file=open(path+"//"+filename)
				email=Email_nltk.Email_nltk(file)
				file.close()
				dictionary=list(set( dictionary + email.get_list()))

			except IOError:
				print("Error: file: "+file_name + " does not exist")
	dictionary.sort()
	print("dictionary: ",len(dictionary))
	return dictionary	

def training_set(dict_path,train_path):	
	path="..//..//lingspam_public//bare//part1"
	spam=0;
	not_spam=0;
	dt=dictionary(dict_path)
	train_X=[]
	train_Y=[]
	for filename in os.listdir(train_path):
		try:
			file=open(train_path+"//"+filename)
			if "spm" in filename:
				spam = spam +1 
			else: 
				not_spam = not_spam + 1 
			email=Email.Email(file)
			file.close()
			train_X.append(email.get_matrix(dt))
			train_Y.append(email.type)
		except IOError:
			print("Error: file: "+file_name + " does not exist")		
	print("spam: ",spam,"not_spam",not_spam)	
	return train_X,train_Y

def spam_set(dict_path,train_path):
	st = dictionary(dict_path)
	spam_X = []
	spam_Y = []
	for filename in os.listdir(train_path):
		if "spm" in filename:
			file=open(train_path+"//"+filename)
			email=Email.Email(file)
			file.close()
			spam_X.append(email.get_matrix(st))
			spam_Y.append(email.type)
	return spam_X,spam_Y

def not_spam_set(dict_path,train_path):
	dt = dictionary(dict_path)
	not_spam_X = []
	not_spam_Y = []
	for filename in os.listdir(train_path):
		if "spm" not in filename:
			file=open(train_path+"//"+filename)
			email=Email.Email(file)
			file.close()
			not_spam_X.append(email.get_matrix(dt))
			not_spam_Y.append(email.type)
	return not_spam_X,not_spam_Y

def training_set_nltk(dict_path,train_path):	
	path="..//..//lingspam_public//bare//part1"
	dt=dictionary_nltk(dict_path)
	train_X=[]
	train_Y=[]
	spam =0
	not_spam = 0
	for filename in os.listdir(train_path):
		try: 
			file=open(train_path+"//"+filename)
			if "spm" in filename:
				spam = spam +1 
			else: 
				not_spam = not_spam + 1 
			email=Email_nltk.Email_nltk(file)
			file.close()
			train_X.append(email.get_matrix(dt))
			train_Y.append(email.type)
		except IOError:
			print("Error: file: "+file_name + " does not exist")

	print("spam: ",spam,"not_spam",not_spam)	
	return train_X,train_Y

def spam_set_nltk(dict_path,train_path):
	st = dictionary_nltk(dict_path)
	spam_X = []
	spam_Y = []
	for filename in os.listdir(train_path):
		if "spm" in filename:
			file=open(train_path+"//"+filename)
			email=Email_nltk.Email_nltk(file)
			file.close()
			spam_X.append(email.get_matrix(st))
			spam_Y.append(email.type)
	return spam_X,spam_Y

def not_spam_set_nltk(dict_path,train_path):
	dt = dictionary_nltk(dict_path)
	not_spam_X = []
	not_spam_Y = []
	for filename in os.listdir(train_path):
		if "spm" not in filename:
			file=open(train_path+"//"+filename)
			email=Email_nltk.Email_nltk(file)
			file.close()
			not_spam_X.append(email.get_matrix(dt))
			not_spam_Y.append(email.type)
	return not_spam_X,not_spam_Y
