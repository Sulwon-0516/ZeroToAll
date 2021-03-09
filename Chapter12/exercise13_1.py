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

import os
import name_data_loader
# This is simple RNN case. no linear layer between them.
INPUT_SIZE = 29
EMBED_SIZE = 20
OUTPUT_SIZE = 18
HIDDEN_SIZE = 100
OHE_DIM = 26 + 3
N_LAYER = 2

BATCH_SIZE = 256
EPOCH = 100
LEARNING_RATE = 0.001

CHECKPOINT_PATH = "./Chapter12/LSTM_model"

class simple_LSTM(nn.Module):
    def __init__(self):
        super(simple_LSTM, self).__init__()
        # nn.embedding and nn.Linear are same.
        self.linear_embed = nn.Linear(in_features = INPUT_SIZE, out_features = EMBED_SIZE)
        self.LSTM = nn.LSTM(input_size = EMBED_SIZE, hidden_size = HIDDEN_SIZE, num_layers = N_LAYER, batch_first=False)
        self.linear_out = nn.Linear(in_features = HIDDEN_SIZE, out_features = OUTPUT_SIZE)

        # If I want to change the input / output size -> need Linear Layers

    def forward(self, input):
        # First resize the input 
        batch_size = input.shape[1]
        # input of shape (seq_len, batch, input_size):
        input = input.view(-1,INPUT_SIZE)
        out = self.linear_embed(input)
        out = out.view(-1,batch_size, EMBED_SIZE)
        # Second run the RNN

        hidden_input = self.init_hidden(batch_size)
        out, hidden = self.LSTM(out, hidden_input)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # output of hidden (num_layers * num_directions, batch, hidden_size)

        # out = out.view(1,-1, HIDDEN_SIZE)
        # I want last block's output.
        # But, what if it's <pad> ? 
        out = self.linear_out(out)
        # out = out.view(1,-1,OUTPUT_SIZE)
        
        return out

    def init_hidden(self,b_size):
        return Variable(torch.randn(N_LAYER,b_size,HIDDEN_SIZE)).float(),Variable(torch.randn(N_LAYER,b_size,HIDDEN_SIZE)).float()





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
    
    
    for epoch in range(EPOCH):        
        tot_loss = 0
        tot_acc = 0
        # i need to push by one character, not one word
        for i, data in enumerate(train_data_loader):
            inputs, labels, lengths = data
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)
            outputs = outputs.view(-1,outputs.shape[-1])
            output = outputs[tuple(lengths),:]

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            
            # calculate the accuracy & total loss
            tot_loss += loss*lengths.shape[0]
            _, prediction = output.max(axis = 1)
            tot_acc += sum(prediction==labels)

            if i%50 == 0:
                print(" loss :",loss)
        tot_loss = tot_loss/train_data.__len__()
        tot_acc = tot_acc/train_data.__len__()
        print("\nepoch : %d, loss : %f, acc : %f" %(epoch+1,tot_loss,tot_acc))
        if epoch%10 == 0:
            torch.save({'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss':loss
            },os.path.join(CHECKPOINT_PATH,(str(BATCH_SIZE)+"_"+str(LEARNING_RATE)+"_"+str(epoch)+".pt")))




        
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


    

        
        