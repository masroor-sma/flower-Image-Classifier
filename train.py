import argparse
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import os, random
import json

from torch import nn, optim
from torchvision import models, datasets, transforms
from PIL import Image
from collections import OrderedDict

def checking_parameters():
    print('Checking Parameters')
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception('--gpu option is enabled but no GPU Detected by the workspace')
    if (not os.path.isdir(args.data_directory)):
        raise Exception('The directory chosen does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test', 'train', 'valid'})):
        raise Exception('The test directory or train directory or the validation directory is missing')
    if args.arch not in ('vgg', 'densenet', None):
        raise Exception('Please choose atleast one model architecture- vgg or densenet, none chosen')
        
def data_preprocessing(data_dir):
    print('Pre-processing data into DataLoaders')
    train_dir, valid_dir, test_dir = data_dir
    
    #Defining the transforms for training, validation and testing
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229,0.224,0.225])])
    
    #loading the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    #Defining the Dataloaders using the datasets
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)
    
    # Label Mapping from the json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    loaders = {'train':trainloaders, 'valid':validloaders, 'test':testloaders, 'labels': cat_to_name}
    
    return loaders

def get_data():
    print('Getting the data directory')
    train_dir = os.path.join(args.data_directory, 'train')
    valid_dir = os.path.join(args.data_directory, 'valid')
    test_dir = os.path.join(args.data_directory, 'test')
    data_dir = [train_dir, test_dir, valid_dir]
    return data_preprocessing(data_dir)

def model_finetuning(data):
    print('building and Finetuning the model')
    if (args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = args.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained = True)
        input_node = 25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained = True)
        input_node=1024
    if (args.hidden_units1 is None):
        hidden_units1 = 2048
    else:
        hidden_units1 = args.hidden_units1
    if (args.hidden_units2 is None):
        hidden_units2 = 256
    else:
        hidden_units2 = args.hidden_units2
    for param in model.parameters():
        param.requires_grad = False
    hidden_units1, hidden_units2 = int(hidden_units1), int(hidden_units2)
    classifier = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(input_node, hidden_units1)),
                                          ('relu1', nn.ReLU()),
                                          ('fc2', nn.Linear(hidden_units1, hidden_units2)),
                                          ('relu2', nn.ReLU()),
                                          ('fc3', nn.Linear(hidden_units2, 102)),
                                          ('output',nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    return model

def check_accuracy_on_test(loader, model, device = 'cpu'):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            criterion = nn.NLLLoss()
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            
            loss += batch_loss.item()
            
            #calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            
    return accuracy/len(loader)

            
    

def train(model, data):
    print('Training Model!')
    
    print_every = 5
    
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    
    if (args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
        
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
        
    learnrate = float(learn_rate)
    epochs = int(epochs)
    
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    
    steps = 0
    running_loss = 0
    model.to(device)
    
    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            #Forward and backward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps%print_every == 0:
                valid_accuracy = check_accuracy_on_test(validloader, model, device)
                
                print('Epochs: {}/{}'.format(e+1, epochs))
                print(f'Training Loss: {running_loss/print_every:.3f}..')
                print(f'Validation Accuracy: {valid_accuracy:.3f}..')
                running_loss = 0
                
    print('Finished Training!!!')
    
    test_result = check_accuracy_on_test(testloader,model,device)
    print('Final Accuracy on Test Set: {}'.format(test_result))
    return model


def save_model(model):
    print('Saving Model')
    if (args.save_dir is None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    checkpoint = {
                   'model':model.cpu(),
                   'features':model.features,
                   'classifier':model.classifier,
                   'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    checking_parameters()
    data = get_data()
    model = model_finetuning(data)
    model = train(model,data)
    save_model(model)
    return None

def parse():
    parser = argparse.ArgumentParser(description = 'Train a neural network with many options!')
    parser.add_argument('data_directory', help = 'data directory(required)')
    parser.add_argument('--save_dir', help = 'directory to save a neural network.')
    parser.add_argument('--arch', help = 'models to use for finetuning(vgg, densenet)', choices=['vgg','densenet'])
    parser.add_argument('--learning_rate','--lr',help = 'learning rate')
    parser.add_argument('--hidden_units1', help = 'number of hidden units in the first layer')
    parser.add_argument('--hidden_units2', help = 'number of hidden units in the second layer')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu', action = 'store_true', help = 'gpu')
    args = parser.parse_args()
    return args

def main():
    print('creating a deep learning model')
    global args
    args = parse()
    create_model()
    print('Model Finished!')
    return None

main()
                        
    
    



        
        
                
    
    
                               
    
        
    
        
        
        
        
    
    
    
    
    
    
    
    
