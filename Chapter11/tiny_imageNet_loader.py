# I will test with data loader in local machine shortly
# and, I will train on GCP.

# tiny imageNet test data doens't have answer label.
# tiny imageNet train & val have several divided folders



import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from torchvision import transforms
import os, os.path
import math
import matplotlib.pyplot as plt
import PIL
from PIL import Image
# Last two things are for image processing

TRAIN_ROOT = "./Dataset/tiny_imageNet/tiny-imagenet-200/train"
VAL_ROOT = "./Dataset/tiny_imageNet/tiny-imagenet-200/val"

# it checks the number of images in each folder
# I checked that ALL TRAIN have 500 images, and ALL TEST have 500 images

def img_num_checker():
    train_length = []
    for folder in os.listdir(TRAIN_ROOT):
        DIR = os.path.join(TRAIN_ROOT,folder)
        num_img = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        #print("train:",num_img)
        train_length.append(num_img)
    val_length = []
    for folder in os.listdir(VAL_ROOT):
        DIR = os.path.join(VAL_ROOT,folder)
        num_img = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        print("val:",num_img)
        val_length.append(num_img)
    
    return train_length, val_length

# The list of the classes are written in winds.txt
# The label of the classes are written in words.txt
# I need to make a tuple of "string"(can be changed as #) <> "string"

def get_class_list():
    f = open("./Dataset/tiny_imageNet/tiny-imagenet-200/wnids.txt",'r')
    labels = [x.strip() for x in f]
    # I referenced this one line method from https://github.com/rmccorm4/Tiny-Imagenet-200/blob/master/networks/data_utils.py
    f.close()
    return labels

def get_class_name(labels):
    # read all data, and search. (it can be done by one-way searching)
    f = open("./Dataset/tiny_imageNet/tiny-imagenet-200/words.txt",'r')
    lines = f.readlines()

    words = dict(line.strip().split('\t') for line in lines)
    # I got the using dict idea from same reference above.
    # I will not split each images when there is several names.
    names = [words[label] for label in labels]
    f.close()
    return names


class tiny_imNet_train_data(Dataset):
    def __init__(self, transforms_in):
        self.labels = get_class_list()
        self.names = get_class_name(self.labels)
        # Apply following transformation to all images, No Augmentation.
        self.transform = transforms_in
        # As I use CSE in torch, I don't need OHE
        '''
        self.OHE = torch.zeros((1,200),dtype=torch.bool)
        self.OHE[0][0] = 1
        for i in range(199):
            temp = torch.zeros((1,200),dtype=torch.bool)
            temp[0][i+1] = 1
            self.OHE = torch.cat((self.OHE,temp), dim = 0)
        print(self.OHE)
        print(self.OHE.size())
        '''

    def __getitem__(self,index):
        # I have to make rule that index into 200 different classes.
        # As all images have same number of images 500, I will use the remainder
        img_ind = math.floor(index/200)
        img_class = index%200

        #print(img_ind,img_class)
        
        DIR = os.path.join(TRAIN_ROOT,self.labels[img_class])
        i = 0
        img_name = '' 
        for name in os.listdir(DIR):
            if i == img_ind:
                img_name = name
                break
            else:
                i = i+1
    
        image = Image.open(os.path.join(DIR,img_name)).convert('RGB')


        # As Inception need 299x299 input, I need to upscale it. 

        # When I test with DenseNet, don't need it.
        #image = image.resize((299,299),Image.ANTIALIAS).convert('RGB')

        # I will apply ImageNet Normalization to the image.
        image = self.transform(image)

        return image, img_class

    def __len__(self):
        #By reducing this number, I can over fit it.
        #original value : 500 * 200
        return 10*200

    def get_name(self, img_class):
        label = self.names[img_class]
        return label

class tiny_imNet_valid_data(Dataset):
    def __init__(self,transforms_in):
        # just cp the above train init.
        super(tiny_imNet_valid_data,self).__init__()
        self.labels = get_class_list()
        self.names = get_class_name(self.labels)
        # Apply following transformation to all images, No Augmentation.
        self.transform = transforms_in

    def __getitem__(self,index):
        # I have to make rule that index into 200 different classes.
        # As all images have same number of images 500, I will use the remainder
        img_ind = math.floor(index/200)
        img_class = index%200

        #print(img_ind,img_class)
        
        DIR = os.path.join(VAL_ROOT,self.labels[img_class])
        i = 0
        img_name = '' 
        for name in os.listdir(DIR):
            if i == img_ind:
                img_name = name
                break
            else:
                i = i+1
    
        image = Image.open(os.path.join(DIR,img_name)).convert('RGB')

        # now, I need the label.
        # label = self.names[img_class]

        # As Inception need 299x299 input, I need to upscale it. 
        #image = image.resize((299,299),Image.ANTIALIAS).convert('RGB')

        # I will apply ImageNet Normalization to the image.
        image = self.transform(image)

        return image, img_class

    def __len__(self):
        return 50*200

    def get_name(self, img_class):
        label = self.names[img_class]
        return label



if __name__ == "__main__":
    _,_ = img_num_checker()

    tester = tiny_imNet_valid_data(transforms.ToTensor())
    A, B = tester.__getitem__(20123)
    print(A)
    print(B)
    