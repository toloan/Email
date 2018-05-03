import os
import re
class Email:
    def __init__(self,file):
        self.source= os.path.realpath(file.name)
        file.seek(0)
        self.subject=(file.readline())[9:]
        if("lingspam_public/bare/part1/3-1msg1.txt" in self.source):
            for i in range(0,6):
                file.readline()
        n=file.tell()
        self.content=(file.read())
        if "spm" in file.name:
            self.type=1
        else:
            self.type=0    

    def get_list(self):
        self.content=self.content.lower()
        words = re.sub(r'[.!,;?"#$%&\'()*+-/@\\]', ' ', self.content).split()
        words=list(set(words))
        words=[word for word in words if word.isdigit()==False]
        return words

    def get_matrix(self,dictionary):
        word_list=self.get_list()
        return [1 if word in word_list else 0 for word in dictionary ]



    
