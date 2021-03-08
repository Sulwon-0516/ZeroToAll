# I need to implement simple "hihell" -> "ihello" model with RNN
# Things to be made
# 1. Text to OHE.
# 2. RNN module (just get it from torch)
# 3. training part.

import numpy as np
import torch 
import torch.nn as nn
import sys

# This is simple RNN case. no linear layer between them.
INPUT_SIZE = 29
OUTPUT_SIZE = 29

LONGEST_WORD = 6
HIDDEN_SIZE = 29
OHE_DIM = 26 + 3
BATCH_SIZE = 1
EPOCH = 200
LEARNING_RATE = 0.1

# One for <pad>, One for <start>, One for <end> and 
# OHE_DIM - 1 : <pad>
# OHE_DIM - 2 : <start>
# OHE_DIM - 3 : <end>


def char_label(word_arr,logest_word = LONGEST_WORD):
    # input dim ( # of words )
    # ouput dim ( # of words ) x (# of characters)
    output = np.ones((1,LONGEST_WORD))
    for i, word in enumerate(word_arr):
        result = np.array([[ord(char) - ord('a') for char in word]])
        for j in range(LONGEST_WORD - len(word)):
            result = np.append(result,[[OHE_DIM-1]],axis=1)
        output = np.append(output,result,axis = 0)
    output = np.delete(output,0,axis=0)

    return output

# I didn't applied the aligning the words by the length of the words. 
# If I want to run the model with high score, I need to add sorting method 

def char_OHE(word_arr,longest_word = LONGEST_WORD):
    # It gets the word array as input (1D) and return the OHE data (3D)
    # input dim : ( # of words )
    # output dim : ( # of words ) x (# of characters) x ( 26 alphabets)

    # Currently assuming the word input is just List.
    result = np.ones((1,longest_word,OHE_DIM))
    for i, word in enumerate(word_arr):
        encoded_word = np.ones((1,1,OHE_DIM))
        num_cnt = LONGEST_WORD
        for char in word:
            encoded_char = np.zeros((1,1,OHE_DIM))
            encoded_char[0][0][ord(char) - ord('a')] = 1
            encoded_word = np.append(encoded_word, encoded_char,axis=1)
            num_cnt = num_cnt -1
        encoded_word = np.delete(encoded_word,0,axis=1)
        # padding with Last words.

        for j in range(num_cnt):
            encoded_char = np.zeros((1,1,OHE_DIM))
            encoded_char[OHE_DIM-1] = 1
            encoded_word = np.append(encoded_word,encoded_char,axis=1)
        
        result = np.append(result, encoded_word,axis=0)
    result = np.delete(result,0,axis=0)

    # Now return the array
    return result

class simple_RNN(nn.Module):
    def __init__(self):
        super(simple_RNN, self).__init__()
        self.rnn = nn.RNN(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, num_layers = 1)

        # If I want to change the input / output size -> need Linear Layers

    def forward(self, input, hidden_input):
        # First resize the input 
        input = input.view(1, BATCH_SIZE , INPUT_SIZE)
        # input of shape (seq_len, batch, input_size):

        # Second run the RNN
        out, hidden = self.rnn(input, hidden_input)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # output of hidden (num_layers * num_directions, batch, hidden_size)
        
        return out,hidden

    def init_hidden(self):
        return torch.autograd.Variable(torch.randn(BATCH_SIZE,1,HIDDEN_SIZE)).float()

def main():
    # Call the data
    # x ; hihell y ; ihello
    x = "hihell"
    y = "ihello"

    inputs = torch.autograd.Variable(torch.tensor(char_OHE([x]))).float()
    labels = torch.autograd.Variable(torch.tensor(char_label([y]))).long()
    

    print(inputs)
    print(labels)

    # traing
    model = simple_RNN()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    

    for epoch in range(EPOCH):
        optimizer.zero_grad()
        hidden = model.init_hidden()
        
        loss = 0
        # i need to push by one character, not one word
        for input, label in zip(inputs,labels):
            print(inputs.shape)
            print(labels.shape)
            for x,y in zip(input,label):
                output, hidden = model(x,hidden)
                # when batch size != 1, there are several outputs.
                result, idx = output.max(2)
                sys.stdout.write(chr(idx+ord('a')))
                loss += criterion(output[0], y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        print("\nepoch : %d, loss : %f" %(epoch+1,loss))

        
if __name__ == '__main__':
    '''
    x = "hihell"
    y = "ihello"

    inputs = char_OHE([x])
    labels = char_label([y])

    print(inputs.shape)
    print(labels)
    '''
    main()


    

        
        