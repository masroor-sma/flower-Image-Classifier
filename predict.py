import argparse
import torch
import numpy as np
import json
import time

from torch import nn , optim
from torchvision import datasets, models, transforms
from PIL import Image

def load_model():
    model_info = torch.load(args.model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def process_image(image):
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    image = img_transforms(img)
    return image


def predict_image(image_path, model, topk=5):
    with torch.no_grad():
        image = process_image(image_path).numpy()
        image = torch.from_numpy(np.array([image])).float()
        model =load_model()
        if (args.gpu):
            image = image.cuda()
            model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
            
            
        logps = model.forward(image)
        probability = torch.exp(logps).topk(topk)
        
    return probability

def read_categories():
    if (args.category_names is not None):
        cat_file = args.category_names
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None

def display_prediction(results):
    cat_file = read_categories()
    probs, classes = results
    probs , classes = probs[0].tolist(), classes[0].add(1).tolist() 
    f_results = zip(probs, classes)
    i = 0
    for p, c in f_results:
        i = i + 1
        p = f'{p*100.:.3f}'+'%'
        if (cat_file):
            c = cat_file.get(str(c),'None')
        else:
            c = ' class {}'.format(str(c))
        print('{}.{} ({})'.format(i,c,p))
        
    return None

def parse():
    parser = argparse.ArgumentParser(description = 'To predict the image using the trained neural network.')
    parser.add_argument('image_input', help='The input image which needs to be classified(required)')
    parser.add_argument('model_checkpoint', help= 'model checkpoint to be used for classification')
    parser.add_argument('--top_k', help = 'how many predictions to be shown to user[default = 5]')
    parser.add_argument('--category_names', help = 'file to be provided if inputting an image from other dataset or different flower species')
    parser.add_argument('--gpu', action='store_true', help='gpu_option')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse()
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception('--gpu option is enabled but no gpu detected')
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.image_input
    prediction = predict_image(image_path,top_k)
    display_prediction(prediction)
    return prediction

main()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    