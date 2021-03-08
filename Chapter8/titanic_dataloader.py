# I will implement simple MLP to classify the datasets.
# I downloaded Titanic Dataset from the Kaggle with Kaggle API

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# autograd Variable will treat the data as trackable.


# About Titanic Dataset
# First line shows the data name. 

# Test : 
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# As Kaggle is closed, this test dataset is useless here.

# Train : 
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C


#-----------------------------------------------------------#
# From Second Line, it contains the data.
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower

# 0 is dead, 1 is survived (True)

# Purpose
# -> Get User Info, and predict him died or not. 

# Which Data I used
# I used Sex, Age, fare and pclass. 
# I wanted to use the Cabin data but as time isn't enough, I gave up.

#I need to call train and test set seperately
class Titanic_test(Dataset):
    def __init__(self):
        test_data = np.loadtxt('./Dataset/titanic/test.csv',delimiter=',',dtype = np.string)
        self.shape = test_data.shape
        
        
    def __getitem__(self,index):
        return 

    def __len__(self):
        return self.shape[0]


class Titanic_train(Dataset):
    def __init__(self):
        # Due to the conbined data types, I call loadtxt many time.
        data = np.genfromtxt('./Dataset/titanic/train.csv',skip_header=1,delimiter=',',dtype = 'str')
        train_data = data
        #what happen to the first line?
        #print(train_data[0:20,(2,4,6)])


        self.shape = train_data.shape
        self.survived = torch.from_numpy(train_data[:,1].astype(np.float)).float()
        
        # [female male], (1,0) : female, (0,1) : male
        sex = np.empty((1,2),dtype=np.float)
        for ind, i in enumerate(train_data[:,5]): 
            sex = np.append(sex,np.array([[0,1]]),axis=0) if i == 'female' else np.append(sex,np.array([[1,0]]),axis=0)
        sex = np.delete(sex,(0,0),axis=0)

        self.sex = torch.from_numpy(np.array(sex).astype(np.float))
        self.fare = torch.from_numpy(train_data[:,10].astype(np.float))
        # [class1 class2 class3] 
        # 1st : [1,0,0], 2nd : [0,1,0], 3rd : [0,0,1]
        pclass = np.empty((1,3),dtype=np.float)
        for ind, i in enumerate(train_data[:,2]):
            if i == 1:
                pclass = np.append(pclass,np.array([[1,0,0]]),axis=0) 
            elif i==2:
                pclass = np.append(pclass,np.array([[0,1,0]]),axis=0)
            else:
                pclass = np.append(pclass,np.array([[0,0,1]]),axis=0)
        pclass = np.delete(pclass,[0,0],axis=0)
        self.pclass = torch.from_numpy(np.array(pclass).astype(np.float))

        # There are some people don't have age information
        # so I skipped it.
        '''
        self.age = torch.from_numpy(train_data[:,6].astype(np.float))
        '''

        self.fare = self.fare.unsqueeze(axis=1)
        # Now concatenate those dataset.
        #print("sex:",self.sex.size())
        #print("fare:",self.fare.size())
        #print("pclass",self.pclass.size())
        
        self.x = torch.cat((self.sex, self.fare, self.pclass),axis = 1)
        self.x = self.x.float()
    
    def __getitem__(self,index):
        return self.x[index], self.survived[index]
    
    def __len__(self):
        return self.shape[0]



if __name__=="__main__":
    dataset = Titanic_train()
    for i in range(10):
        inputs, labels = dataset.__getitem__(i)
        print("data: ",inputs,"label: ",labels)

# Things I should care about : 
# Should I normalize or scale it into [0,1] ? 
# -> I have boolean type inputs, so scaling would be helpful.
# Then, The initial bias on the pclass can increase? 
# -> Not at all.... I think normalizing the fare and age is also good method.


# 2021.Mar.2 Checked working correctly.