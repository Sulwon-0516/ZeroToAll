import titanic_dataloader as C
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

FIRST_INPUT_FEATURE = 6
FIRST_NEURON = 30
SECOND_NUERON = 30
OUTPUT = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCH = 50


#Define simple 3-Depth MLP
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.L1 = nn.Linear(in_features = FIRST_INPUT_FEATURE, out_features = FIRST_NEURON)
        self.L2 = nn.Linear(FIRST_NEURON, SECOND_NUERON)
        self.L3 = nn.Linear(SECOND_NUERON,OUTPUT)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.tanh(self.L1(x))
        out = self.tanh(self.L2(out))
        out = self.sigmoid(self.L3(out))
        return out




def main():
    dataset = C.Titanic_train()
    train_loader = DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    model = Classifier()

    #I used BCE and adama optimizer 
    criterion = nn.BCELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
    # betas, weight_decay, eps are skipped.


    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            prediction = model(inputs)
            '''
            for j, data in enumerate(prediction):
                if data < 0.5:
                    prediction[i] = 0
                else:
                    prediction[i] = 1
            '''
            loss = criterion(prediction,labels.unsqueeze(axis=1))

            #Calculate the batch accuracy
            acc = 0
            for j, data in enumerate(prediction):
                if data < 0.5 and labels[j] == 0:
                    acc = acc+1
                elif data>0.5 and labels[j] == 1:
                    acc = acc+1
            acc = acc/BATCH_SIZE
            if epoch%5 == 0 :
                print("epoch:",epoch,"," ,i,",loss : ", loss.item(),",acc : ",acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()