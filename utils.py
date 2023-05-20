import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import torch 
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json 
import os

def plot(data,ylb,title):
    plt.plot([i for i in range(1,)],data,color='mediumvioletred',marker='o')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(ylb)
    plt.savefig(f'results/{title}.png')

def crop_resize(img,dim=224):
    img=np.array(img)
    pil_image1=Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize(dim),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    return transform(pil_image1)

def crop_resize_cv2(img,dim=128):
    sz=(dim,dim)
    img_resized=cv2.resize(img,sz,interpolation = cv2.INTER_AREA)
    return img_resized

def phi(state):
    state_vis=state["visual"]
    state["visual"]=np.array(crop_resize(state_vis,dim=224))

def phiXtra(state):
    state_vis=state["visual"]
    state["visual"]=np.array(crop_resize_cv2(state_vis,dim=128))
    state["visual"]=np.mean(state["visual"],axis=2) # Reduce channel

def oneHot(mx,x):
    return torch.tensor([0 if i!=x else 1 for i in range(mx)]).unsqueeze(0)

unique_name={}
name_len=10
def save(state):
    new_name=""
    while True:
        new_name_arr=[ chr(ord('a')+np.random.randint(0,name_len)) for x in range(name_len) ]
        for ch in new_name_arr:
            new_name+=ch
        if new_name not in unique_name:
            unique_name[new_name]=True 
            break 
    with open(f"save_state/{new_name}.json",'w') as f:
        state["visual"]=state["visual"].tolist()
        json.dump(state,f)
    return new_name

def delete(state):
    del unique_name[state]

def get(state):
    f=open(f'save_state/{state}.json')
    data=json.load(f)
    state=data
    state["visual"]=np.array(state["visual"])
    return state