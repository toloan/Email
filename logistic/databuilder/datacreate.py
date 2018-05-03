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
	print(email.get_matrix(dictionary))
	return dictionary	

def training_set(dict_path,train_path):	
	path="..//..//lingspam_public//bare//part1"
	dictionary=dictionary(dict_path)
	train_X=[]
	train_Y=[]
	for filename in os.listdir(train_path):
		file=open(path+"//"+filename)
		email=Email.Email(file)
		file.close()
		train_X.append(email.get_matrix(dictionary))
		train_Y.append(email.type)
	return train_X,train_Y

