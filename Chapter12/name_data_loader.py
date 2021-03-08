import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


INPUT_SIZE = 29
EMBED_SIZE = 20
OUTPUT_SIZE = 29
HIDDEN_SIZE = 5
OHE_DIM = 26 + 3

TRAIN_DATA_PATH = "./Dataset/name_classification/names_train.csv"
TEST_DATA_PATH = "./Dataset/name_classification/names_test.csv"
# One for <pad>, One for <start>, One for <end> and 
# OHE_DIM - 1 : <pad>
# OHE_DIM - 2 : space
# OHE_DIM - 3 : '

# DataLoader    
# I skipped dividing Validation and Training.
class name_train_data(Dataset):
    def __init__(self):
        data = np.genfromtxt(TRAIN_DATA_PATH, skip_header=0,delimiter='"',dtype = 'str')
        self.data = data[:,1]
        self.label = data[:,3]
        self.n_country = len(set(self.label))

        self.cnt2int, self.encoded_country = country_label(self.label)
        

    def __getitem__(self,index):
        x = char_OHE(self.data[index])
        y = self.encoded_country[index]
        return x, y

    def __len__(self):
        return self.data.shape[0]
        
class name_test_data(Dataset):
    def __init__(self,cnt2int_in,n_country_in):
        data = np.genfromtxt(TEST_DATA_PATH, skip_header=0,delimiter='"',dtype = 'str')
        self.data = data[:,1]
        self.label = data[:,3]
        self.n_country = n_country_in
        self.cnt2int = cnt2int_in

        _, self.encoded_country = country_label(self.label,self.cnt2int)
        

    def __getitem__(self,index):
        x = char_OHE(self.data[index])
        y = self.encoded_country[index]
        return x, y

    def __len__(self):
        return self.data.shape[0]

def country_label(cnt_arr, cnt2int = None):
    # input dim ( # of countries )
    # ouput dim ( # of countries )
    label = set(cnt_arr)
    if cnt2int == None:
        print("cnt2int contruction...")
        cnt2int = {name:i for i,name in enumerate(label)}
    else:
        print(cnt2int)
    
    result = np.zeros(len(cnt_arr))
    for i, word in enumerate(cnt_arr):
        result[i] = cnt2int[word]
    #print(result)

    return cnt2int, result

# I didn't applied the aligning the words by the length of the words. 
# If I want to run the model with high score, I need to add sorting method 

def collate_fn(batches):
    max_length = 0
    for name, label in batches:
        if(len(name) > max_length):
            max_length = len(name[0][0])

    batch_name = np.zeros((max_length,1,OHE_DIM))
    batch_label = np.zeros(1)
    for name, label in batches:
        for j in range(max_length - len(name)):
            pad_token = np.zeros((1,1,OHE_DIM))
            pad_token[0][0][OHE_DIM-1] = 1
            name = np.append(name,pad_token,axis=0)
        batch_name = np.append(batch_name,name,axis=1)
        batch_label = np.append(batch_label,[label],axis=0)
    batch_name = torch.from_numpy(np.delete(batch_name,0,axis=1)).float()
    batch_label = torch.from_numpy(np.delete(batch_label,0,axis=0)).long()


    return batch_name, batch_label






def char_OHE(word):
    # It gets the word array as input (1D) and return the OHE data (3D)
    # input dim : 1 word
    # output dim : 1 x (# of characters) x ( 26 alphabets )

    word = word.lower()
    # Currently assuming the word input is just List.
    encoded_word = np.ones((1,1,OHE_DIM))
    for char in word:
        encoded_char = np.zeros((1,1,OHE_DIM))
        if char == "'":
            encoded_char[0][0][OHE_DIM-3] = 1
        elif char == " ":
            encoded_char[0][0][OHE_DIM-2] = 1
        else:
            encoded_char[0][0][ord(char) - ord('a')] = 1
        encoded_word = np.append(encoded_word, encoded_char,axis=0)
    encoded_word = np.delete(encoded_word,0,axis=0)
    
    # Now return the array
    return encoded_word


if __name__ == '__main__':
    A = name_train_data()

    train_loader = DataLoader(dataset = A, batch_size = 16, shuffle = True,collate_fn = collate_fn)

    for i, data in enumerate(train_loader):
        if i == 1:
            inputs, labels = data
            print(inputs)
            print(inputs.shape)
            print(labels)
