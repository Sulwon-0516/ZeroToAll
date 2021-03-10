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
from torchsummary import summary



import os
import name_data_loader
# This is simple RNN case. no linear layer between them.
INPUT_SIZE = 29
EMBED_SIZE = 20 
OUTPUT_SIZE = 18
HIDDEN_SIZE = 200
OHE_DIM = 26 + 3
N_LAYER = 3

BATCH_SIZE = 256
EPOCH = 200
LEARNING_RATE = 2e-4

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
        
        out = self.linear_embed(input)
        # out = out.view(-1,batch_size, EMBED_SIZE)
        # Second run the RNN

        hidden = self.init_hidden(batch_size)
        out, hidden = self.LSTM(out, hidden)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # output of hidden (num_layers * num_directions, batch, hidden_size)
       

        # out = out.view(1,-1, HIDDEN_SIZE)
        # I want last block's output.
        # But, what if it's <pad> ? 

        out = self.linear_out(hidden[0][-1,:,:])
        # out = out.view(1,-1,OUTPUT_SIZE)

        return out

    def init_hidden(self,b_size):
        return Variable(torch.randn(N_LAYER,b_size,HIDDEN_SIZE)).float(),Variable(torch.randn(N_LAYER,b_size,HIDDEN_SIZE)).float()



def main():
    # Call the data

    train_data = name_data_loader.name_train_data()
    tot_size = train_data.__len__()

    train_data_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = False, collate_fn = name_data_loader.collate_fn)

    # train
    model = simple_LSTM()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = LEARNING_RATE, alpha = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [40, 60], gamma = 0.2)
    for epoch in range(EPOCH):        
        tot_loss = 0
        tot_acc = 0
        # i need to push by one character, not one word
        for i, data in enumerate(train_data_loader):
            inputs, labels, lengths = data
            inputs, labels = Variable(inputs), Variable(labels)


            output = model(inputs)
            # outputs = model(inputs)
            # outputs = outputs.view(-1,outputs.shape[-1])
            # output = outputs[tuple(lengths),:]

            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            
            # calculate the accuracy & total loss
            tot_loss += loss*lengths.shape[0]
            _, prediction = output.max(axis = 1)
            tot_acc += sum(prediction==labels)

            
            if i%20 ==0 :
                #print(output[0:10,:])
                #print(prediction[1:20])
                #print(labels[1:20])
                #A=input()
                print(" loss :",loss)
        print(tot_acc)
        scheduler.step()
        tot_loss = tot_loss/tot_size
        tot_acc = tot_acc/tot_size
        print("epoch : %d, loss : %f, acc : %f" %(epoch+1,tot_loss,tot_acc))
        if epoch%10 == 0:
            torch.save({'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss':loss
            },os.path.join(CHECKPOINT_PATH,(str(BATCH_SIZE)+"_"+str(LEARNING_RATE)+"_"+str(epoch)+"-1.pt")))

def retrain():

    train_data = name_data_loader.name_train_data()
    tot_size = train_data.__len__()

    model = simple_LSTM()
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,"256_0.0002_40-1.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_data_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = False, collate_fn = name_data_loader.collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = LEARNING_RATE, alpha = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [40, 60], gamma = 0.2)
    for epoch in range(EPOCH):        
        tot_loss = 0
        tot_acc = 0
        # i need to push by one character, not one word
        for i, data in enumerate(train_data_loader):
            inputs, labels, lengths = data
            inputs, labels = Variable(inputs), Variable(labels)


            output = model(inputs)
            # outputs = model(inputs)
            # outputs = outputs.view(-1,outputs.shape[-1])
            # output = outputs[tuple(lengths),:]

            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            
            # calculate the accuracy & total loss
            tot_loss += loss*lengths.shape[0]
            _, prediction = output.max(axis = 1)
            tot_acc += sum(prediction==labels)

            
            if i%20 ==0 :
                #print(output[0:10,:])
                #print(prediction[1:20])
                #print(labels[1:20])
                #A=input()
                print(" loss :",loss)
        print(tot_acc)
        scheduler.step()
        tot_loss = tot_loss/tot_size
        tot_acc = tot_acc/tot_size
        print("epoch : %d, loss : %f, acc : %f" %(epoch+1,tot_loss,tot_acc))
        if epoch%10 == 0:
            torch.save({'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss':loss
            },os.path.join(CHECKPOINT_PATH,("retrain_"+str(BATCH_SIZE)+"_"+str(LEARNING_RATE)+"_"+str(epoch)+"-1.pt")))

def test():
    train_data = name_data_loader.name_train_data()
    test_data = name_data_loader.name_test_data(train_data.cnt2int,train_data.n_country)
    test_data_loader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = False, collate_fn = name_data_loader.collate_fn)
    model = simple_LSTM()
    print(model)


    with torch.no_grad():
        
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,"retrain_256_0.0002_30-1.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        
        criterion = nn.CrossEntropyLoss()

        tot_len = test_data.__len__()
        tot_acc = 0
        tot_loss = 0
        for i, data in enumerate(test_data_loader):
            inputs, labels, lengths = data
            inputs, labels = Variable(inputs), Variable(labels)

            output = model(inputs)

            loss = criterion(output,labels)

            tot_loss += loss*lengths.shape[0]
            _, prediction = output.max(axis = 1)
            tot_acc += sum(prediction==labels)

            


def binary_main():

    data = name_data_loader.spanish_polish_data()


    dataLoader = DataLoader(dataset = data, batch_size = 32, shuffle = False, collate_fn = name_data_loader.collate_fn)

    model = simple_LSTM()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 20)


    for epoch in range(EPOCH):        
        tot_loss = 0
        tot_acc = 0
        # i need to push by one character, not one word
        for i, data in enumerate(dataLoader):
            inputs, labels, lengths= data
            inputs, labels = Variable(inputs), Variable(labels)

            
            output = model(inputs)
            # outputs = model(inputs)
            # outputs = outputs.view(-1,outputs.shape[-1])
            # output = outputs[tuple(lengths),:]
           
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            scheduler.step(loss)

            # calculate the accuracy & total loss
            tot_loss += loss*lengths.shape[0]
            _, prediction = output.max(axis = 1)
            acc = sum(prediction==labels)
            tot_acc += acc
            #print (i,acc)
            #print(prediction)
        #print(tot_acc/396)
        tot_loss = tot_loss/396
        tot_acc = tot_acc/396
        print("\nepoch : %d, loss : %f, acc : %f" %(epoch+1,tot_loss,tot_acc))

        
if __name__ == '__main__':
    '''
    x = "hihell"
    y = "ihello"

    inputs = char_OHE([x])
    labels = char_label([y])

    print(inputs.shape)
    print(labels)
    '''
    test()


    

        
        