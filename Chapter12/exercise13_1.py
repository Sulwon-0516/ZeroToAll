# I need to implement simple "hihell" -> "ihello" model with RNN
# Things to be made
# 1. Text to OHE.
# 2. RNN module (just get it from torch)
# 3. training part.

import numpy as np
import torch 
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import name_data_loader
# This is simple RNN case. no linear layer between them.
INPUT_SIZE = 29
EMBED_SIZE = 10
OUTPUT_SIZE = 18
HIDDEN_SIZE = 7
OHE_DIM = 26 + 3

BATCH_SIZE = 1
EPOCH = 200
LEARNING_RATE = 0.01


class simple_LSTM(nn.Module):
    def __init__(self):
        super(simple_LSTM, self).__init__()
        # nn.embedding and nn.Linear are same.
        self.linear_embed = nn.Linear(in_features = INPUT_SIZE, out_features = EMBED_SIZE)
        self.LSTM = nn.LSTM(input_size = EMBED_SIZE, hidden_size = HIDDEN_SIZE, num_layers = 1)
        self.linear_out = nn.Linear(in_features = HIDDEN_SIZE, out_features = OUTPUT_SIZE)

        # If I want to change the input / output size -> need Linear Layers

    def forward(self, input, hidden_input):
        # First resize the input 
        input = input.view(-1, INPUT_SIZE)
        # input of shape (seq_len, batch, input_size):

        out = self.linear_embed(input)

        out = out.view(1,-1, EMBED_SIZE)
        # Second run the RNN

        print(out.shape)
        out, hidden = self.LSTM(out, hidden_input)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # output of hidden (num_layers * num_directions, batch, hidden_size)

        # out = out.view(1,-1, HIDDEN_SIZE)
        out = self.linear_out(out)
        # out = out.view(1,-1,OUTPUT_SIZE)
        
        return out,hidden

    def init_hidden(self):
        return Variable(torch.randn(1,BATCH_SIZE,HIDDEN_SIZE)).float(),Variable(torch.randn(1,BATCH_SIZE,HIDDEN_SIZE)).float()

def main():
    # Call the data

    train_data = name_data_loader.name_train_data()
    test_data = name_data_loader.name_test_data(train_data.cnt2int,train_data.n_country)

    train_data_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True, collate_fn = name_data_loader.collate_fn)
    test_data_loader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = True, collate_fn = name_data_loader.collate_fn)

    # train
    model = simple_LSTM()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(EPOCH):
            hidden = model.init_hidden()
            
            loss = 0
            # i need to push by one character, not one word
            for i, data in enumerate(train_data_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                for j in range(inputs.shape[0]):
                    output, hidden = model(inputs[j,:,:],hidden)
                
                loss = criterion(output[0], labels)
                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()
                print(i)
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


    

        
        