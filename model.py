import torch 
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import seaborn 
import utils as U
import json
import copy 

noOfActions=6
epochA=0
class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18
        self.resnet=models.resnet18(pretrained=True)
        num_dim_in=self.resnet.fc.in_features
        num_dim_out=1024
        self.resnet.fc=nn.Linear(num_dim_in,num_dim_out)
        self.important_features_image=512
        self.fcResNet1=nn.Sequential(
            nn.Linear(num_dim_out,self.important_features_image),
            nn.ReLU()
        )
        for param in self.resnet.parameters():
            param.requires_grad=False 
        layerNotTrainable='layer4'
        for i, (name, param) in enumerate(self.resnet.named_parameters()):
            if name[:len(layerNotTrainable)]=='layer4':
                param.requires_grad=True
        # DQN
        self.noOfActions=6
        self.actor_critic=nn.Sequential(
            nn.Linear(self.important_features_image,self.important_features_image//2),
            nn.ReLU(),
            nn.Linear(self.important_features_image//2,self.noOfActions)
        )
    def forward(self,image,action):
        op=self.resnet(image)
        # opR0=self.fcResNet0(op) # image 
        opR1=self.fcResNet1(op)
        if torch.cuda.is_available():
            action=action.to(torch.device("cuda:0")) # CUDA:0 is the default cuda device in pytorch.
            opR1=opR1.to(torch.device("cuda:0")) 
        q=self.dqn(torch.cat([action,opR1],dim=1))
        # return opR0,q
        return q

    def display(self):
        for param in self.fcResNet0:
            print(param)
'''
------------------------------------------------------------------------------------
'''
def lossDqn(p,y):
    return torch.mean((p-y)**2)

def train(model,reward_true,STATE,ACTION,device,optim,type='dqn',verbose=False):
    global epochA
    # IF YOU NEED TO ENABLE IMAGE TRAINING REMOVE THE FOLLOWING IF CONDITION.
    if type=='image':
        return 
    model.train()
    IMG_X=np.array([_state_["visual"] for _state_ in STATE])
    IMG_X=torch.from_numpy(IMG_X)
    IMG_X=IMG_X.to(device)
    ACTION=ACTION.to(device)
    reward_true=torch.tensor(reward_true)
    # IMG_YP,Q=model(IMG_X.float(),ACTION)
    Q=model(IMG_X.float(),ACTION)
    loss_dqn,loss_image=-1,-1
    if type=='dqn':
        optim.zero_grad()
        loss_dqn=lossDqn(Q,reward_true)
        loss_dqn.backward()
        optim.step()
    if (epochA//2) % 50 == 0 or verbose:
        print(f'Train Epoch:{epochA} DQN_Loss:{loss_dqn} IMG_Loss:{loss_image}')
    epochA+=1

def predict(model,STATE,device,verbose=False):
    global noOfActions
    model.eval()
    with torch.no_grad():
        IMG=np.array([_state_["visual"] for _state_ in STATE])
        IMG=torch.from_numpy(IMG)
        IMG=IMG.to(device)
        QA=torch.empty(IMG.shape[0],0)
        for actions in range(noOfActions):
            action_temp=U.oneHot(noOfActions,actions)
            action_temp=action_temp.repeat(IMG.shape[0],1)
            #IMG_YP,Q=model(IMG.float(),action_temp)
            Q=model(IMG.float(),action_temp)
            if torch.cuda.is_available():
                Q=Q.to(torch.device("cuda:0")) 
                QA=QA.to(torch.device("cuda:0")) 
            QA=torch.cat((QA,Q),dim=1)
        return torch.max(QA,dim=1),torch.argmax(QA,dim=1) 


def dbg1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=NNModel().to(device)
    f=open("my_dict.json")
    data=json.load(f)
    state=data
    state["visual"]=np.array(state["visual"])
    action=(predict(model,[state,state],device))[1]
    print(f'The action to be taken: {action}')

def dbg2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=NNModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    f=open("my_dict.json")
    data=json.load(f)
    state=data
    state["visual"]=np.array(state["visual"])
    temp_next_state=copy.deepcopy(state)
    U.phiXtra(temp_next_state)
    batch_size=2
    STATE=[state for _ in range(batch_size)]
    SMALL_NEXT_STATE=[temp_next_state for _ in range(batch_size)]
    REWARD=np.array([1 for _ in range(batch_size)])
    ACTION=torch.stack([U.oneHot(6,1).squeeze() for _ in range(batch_size)])
    gamma=0.9
    reward_true=REWARD+gamma*((predict(model,STATE,device))[0]).values.numpy()
    train(model,reward_true,STATE,SMALL_NEXT_STATE,ACTION,device,optimizer,type='image',verbose=True)
    train(model,reward_true,STATE,SMALL_NEXT_STATE,ACTION,device,optimizer,type='dqn',verbose=True)


if __name__=='__main__':
    # Preq: Generate my_dict.json from main.ipynb
    #dbg1() -> predict. 
    #dbg2()
    pass