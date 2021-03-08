# Let's try to use torchvision DataLoader in this time.
# It seems hard to split the train/val in torchvision package manually, so skipped here.
# It can be done by manually defining own dataLoader and using random_split function

import wandb
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
import model_test
import exercise10_1

#Parameters 
DATA_PATH = "./Dataset/cifar10"
BEST_CHECKPOINT_PATH = "./Chapter10/model/best_chk_1.pt"
BATCH_SIZE = 64
LEARNING_RATE = 0.005
EPOCH = 0


# Data Augmentation
# First, without image normalization
train_trans = transforms.Compose(
    [
        transforms.RandomCrop(32,padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

test_trans = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

def main():
    #Load data
    #Torchvision downloads the data ONLY if there isn't data
    Path(DATA_PATH).mkdir(exist_ok=True)
    Path("./Chapter10/model").mkdir(exist_ok=True)
    train_data = datasets.CIFAR10(root = DATA_PATH, train = True, transform=train_trans, download=True)
    test_data = datasets.CIFAR10(root = DATA_PATH, train = True, transform=test_trans, download=True)

    train_loader = DataLoader(dataset = train_data, batch_size= BATCH_SIZE, shuffle= True)
    test_loader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle= True)

    
    model = exercise10_1.exercise_CNN()
    #I used Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.3)
    # betas, weight_decay, eps are skipped.

    highest_acc = 0
    for epoch in range(EPOCH):
        acc = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            prediction = model(inputs)
            loss = criterion(prediction,labels)
            pred_class = torch.argmax(prediction, axis = 1)
            acc = acc + torch.sum(pred_class==labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #Calculate the epoch accuracy
        
        acc = acc/train_data.__len__()
        if acc > highest_acc:
            highest_acc = acc
            torch.save({'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss':loss
            },BEST_CHECKPOINT_PATH)
        print("epoch:",epoch,"," ,i,",loss : ", loss.item(),",acc : ",acc,",highest_acc:",highest_acc)

        #Step the scheduler
        scheduler.step()
    
    # Result of the training
    with torch.no_grad():
        model = exercise10_1.exercise_CNN()
        checkpoint = torch.load(BEST_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tot_acc = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            

            prediction = model(inputs)
            pred_class = torch.argmax(prediction, axis = 1)
            acc = torch.sum(pred_class == labels)
            tot_acc = tot_acc + acc
            if i==0:
                print(prediction)
        print("acc :", tot_acc/test_data.__len__())



if __name__ == "__main__":
    main()