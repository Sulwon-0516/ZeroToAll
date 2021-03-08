# Let's try to use torchvision DataLoader in this time.
# It seems hard to split the train/val in torchvision package manually, so skipped here.
# It can be done by manually defining own dataLoader and using random_split function

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
import tiny_imageNet_loader
import exercise11_3
import os
from pathlib import Path

#Parameters 
DATAPATH = "./Dataset/ImageNet"
CHECKPOINT_PATH = "./Chapter11/model/checkpoints/DenseNet"
BEST_CHECKPOINT_PATH = "./Chapter11/model/checkpoints/DenseNet/best.pt"
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EPOCH = 0
kargs = {"k" : [12,12,14,14], "L" : [6,12,24,16], "n_class" : 200}

# Just simple Flipping
# First, without image normalization
train_trans = transforms.Compose(
    [
        #transforms.RandomCrop(32,padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ]
)

val_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ]
)

def main():
    #Load data
    train_data = tiny_imageNet_loader.tiny_imNet_train_data(train_trans)
    val_data = tiny_imageNet_loader.tiny_imNet_valid_data(val_trans)

    train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset = val_data, batch_size = BATCH_SIZE, shuffle = False)
    # In GCP, I will use following torchvision DataLoader
    #Torchvision downloads the data ONLY if there isn't data
    '''
    Path(DATA_PATH).mkdir(exist_ok=True)
    Path("./Chapter10/model").mkdir(exist_ok=True)
    train_data = datasets.ImageNet(root = DATA_PATH, split = 'train', transform=train_trans, download=True)
    valid_data = datasets.ImageNet(root = DATA_PATH, split = 'val', transform=val_trans, download=True)

    train_loader = DataLoader(dataset = train_data, batch_size= BATCH_SIZE, shuffle= True)
    valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE, shuffle= True)

    '''
    Path(CHECKPOINT_PATH).mkdir(exist_ok=True)
    

    model = exercise11_3.DenseNet(**kargs)
    
    # From Inception v4 Paper....
    # Train method of best model.
    # RMSProp with decay of 0.9 and ε = 1.0. 
    # We used a learning rate of 0.045, decayed every two epochs using an exponential rate of 0.94.


    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE, momentum = 0.9, dampening = 0, weight_decay = 0.0001, nesterov = True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [EPOCH/2, EPOCH*3/4], gamma = 0.2, last_epoch = -1)
    # betas, weight_decay, eps are skipped.
    # this is for the log.
    log_f = open(os.path.join(CHECKPOINT_PATH,(str(BATCH_SIZE)+"_"+str(LEARNING_RATE)+".txt")),"w")

    highest_acc = 0
    for epoch in range(EPOCH):
        acc = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            prediction = model(inputs)
            loss = criterion(prediction,labels)
            pred_class = torch.argmax(prediction, axis = 1)
            step_acc = torch.sum(pred_class==labels)
            acc = acc + step_acc

            '''
            if i==0:
                print(prediction)
                print(labels)
            '''

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                log_line = "epoch [%d,%d], loss : %f, step_acc : %f\n" % (epoch,EPOCH,loss,step_acc/labels.size(0))
                log_f.write(log_line)
                print(log_line,end="")
        #Calculate the epoch accuracy
        
        acc = acc/train_data.__len__()
        if acc > highest_acc:
            highest_acc = acc
            torch.save({'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss':loss
            },os.path.join(CHECKPOINT_PATH,"best.pt"))
        print("epoch:",epoch,"," ,i,",loss : ", loss.item(),",acc : ",acc,",highest_acc:",highest_acc)


        if epoch % 10 == 0 and epoch!=0:
            torch.save({'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss':loss
            },os.path.join(CHECKPOINT_PATH,(str(BATCH_SIZE)+"_"+str(LEARNING_RATE)+"_"+str(epoch)+".pt")))
        #Step the scheduler
        scheduler.step()
    
    # Result of the training
    with torch.no_grad():
        model = exercise11_3.DenseNet(**kargs)
        checkpoint = torch.load(BEST_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tot_acc = 0
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            prediction = model(inputs)
            pred_class = torch.argmax(prediction, axis = 1)
            acc = torch.sum(pred_class == labels)
            tot_acc = tot_acc + acc
        print("acc :", tot_acc/val_data.__len__())

    f.close()


if __name__ == "__main__":
    main()