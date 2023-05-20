import gym
from gym import spaces
import pygame
import numpy as np
from enum import Enum 
import vision_pyGame as vga
'''
ACTIONS
'''
class ACTION(Enum):
    PICK_BIG=0
    PICK_MED=1
    PICK_SMALL=2
    PUT_BIG=3
    PUT_MED=4
    PUT_SMALL=5

'''
BOX TYPE
'''
class BOXTYPE(Enum):
    NONE=0
    BIG=1
    MEDIUM=2
    SMALL=3

class RlNlpWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
############################################
    def __init__(self,mx_timeSteps=50,render_mode=None):
        self.mode=0
        if render_mode=='rgb_array':
            self.mode=1
        self._visual=None
        self._text='#'
        self._question='?'
        self.mx_timeSteps,self.curr_time=mx_timeSteps,0
        # Just to remove assertion error. #
        self.action_space = spaces.Discrete(6) 
        self.observation_space=spaces.Dict({
            "text": spaces.Text(min_length=1,max_length=100),"question":spaces.Text(min_length=1,max_length=100),
            "visual": spaces.Box(low=0, high=255, shape=(vga.WIDTH,vga.HEIGHT,3), dtype=np.uint8)
        })
        self.spell={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',0:'zero'}
############################################
    def _get_obs(self):
        return {"text": self._text,"question":self._question,"visual": self._visual}

    def _get_info(self):
        return {
            "progress": "currently feature not required."
        }
############################################
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.no=np.random.randint(1,1000)
        ## Gen initial info ##
        self.carry=False
        self.blocksLeft=[ (self.no//10**i)%10 for i in range(3) ]
        self.blocksLeft.reverse()
        self.boxType=BOXTYPE.NONE
        self.curr_time=0
        self._visual=vga.draw_main(self.metadata['render_modes'][self.mode],self.metadata['render_fps'],self.no)
        self._text=f'Start the experiment. {self.getNLP()}'
        return self._get_obs()
############################################
    def getNLP(self):
        def nlp_util():
            str='There are '
            if self.blocksLeft[0]>0:
                str=str+f'{self.spell[self.blocksLeft[0]]} big blocks left '
            if self.blocksLeft[1]>0:
                str=str+f'{self.spell[self.blocksLeft[1]]} medium blocks left '
            if self.blocksLeft[2]>0:
                str=str+f'{self.spell[self.blocksLeft[2]]} small blocks left '
            return str
        if self.carry==True:
            if self.boxType==BOXTYPE.BIG:
                putBox="big"
                self.exp_action=[3]
            elif self.boxType==BOXTYPE.MEDIUM:
                putBox="medium"
                self.exp_action=[4]
            else:
                putBox="small"
                self.exp_action=[5]
            self._text=f'Put the {putBox} box in the {putBox} digit area.'
        else:
            self._text=f'{nlp_util()}, so you can either pick '
            self.exp_action=[]
            if self.blocksLeft[0]>0:
                self._text=self._text+f'one big block '
                self.exp_action.append(0)
            if self.blocksLeft[1]>0:
                self._text=self._text+f'one medium block '
                self.exp_action.append(1)
            if self.blocksLeft[2]>0:
                self._text=self._text+f'one small block '
                self.exp_action.append(2)
            self._text=self._text+'.'
        return self._text
############################################
    def step(self, action):

        def pick(boxArr,b_type):
            if self.carry==False:
                for box in boxArr:
                    if not box.isEmpty:
                        self.carry=True 
                        self.boxType=b_type
                        box.isEmpty=True
                        return 0 
            else:
                return -1
            
        def put(b_type):
            if self.carry==False or self.boxType!=b_type:
                return -1
            self.boxType=BOXTYPE.NONE 
            self.carry=False 
            # constructArrElement is where the blocks are put. 
            self.blocksLeft[b_type.value-1]-=1
            for box in vga.constructArrElement[b_type.value-1]:
                if box.isEmpty:
                    box.isEmpty=False 
                    return 0 
                
        def checkSolution():
            result,power=0,100
            for c in vga.constructArrElement:
                cnt_box=0
                for box in c:
                    if not box.isEmpty:
                        cnt_box+=1
                result+=cnt_box*power 
                power/=10
            return True if self.no==result else False 
        
        reward=0
        if action==ACTION.PICK_BIG.value:
            reward=pick(vga.big_block,BOXTYPE.BIG)
        elif action==ACTION.PICK_MED.value:
            reward=pick(vga.medium_block,BOXTYPE.MEDIUM)
        elif action==ACTION.PICK_SMALL.value:
            reward=pick(vga.small_block,BOXTYPE.SMALL)
        elif action==ACTION.PUT_BIG.value:
            reward=put(BOXTYPE.BIG)
        elif action==ACTION.PUT_MED.value:
            reward=put(BOXTYPE.MEDIUM)
        elif action==ACTION.PUT_SMALL.value:
            reward=put(BOXTYPE.SMALL)

        reward= 1 if action in self.exp_action else reward
        self._visual=vga.drawAgain()
        self._text=self.getNLP()
        self.curr_time+=1
        solution=checkSolution() # return True is solution is correct
        terminated=False 
        if self.curr_time>self.mx_timeSteps or solution==True:
            terminated=True
        if terminated:
            sign=1 if checkSolution() else -1
            self.close()
        reward = sign*10 if terminated else reward
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, info
############################################
    def render(self):
        pass 
############################################
    def _render_frame(self):
        pass
############################################
    def close(self):
        vga.close_pyame()
############################################