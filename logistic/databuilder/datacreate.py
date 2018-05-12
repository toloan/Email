import os
import Email
def dictionary(path):
	
	dictionary=list()
	for filename in os.listdir(path):
		file=open(path+"//"+filename)
		email=Email.Email(file)
		file.close()
		dictionary=list(set(dictionary+email.get_list()))
	dictionary.sort()
#	print(email.get_matrix(dictionary))
	return dictionary	

def training_set(dict_path,train_path):	
	path="..//..//lingspam_public//bare//part1"
	dt=dictionary(dict_path)
	train_X=[]
	train_Y=[]
	for filename in os.listdir(train_path):
		file=open(train_path+"//"+filename)
		email=Email.Email(file)
		file.close()
		train_X.append(email.get_matrix(dt))
		train_Y.append(email.type)
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
