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

spell={1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth'}
class RlNlpWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000000}
############################################
    def __init__(self,render_mode=None):
        self.mode=0
        if render_mode=='rgb_array':
            self.mode=1
        self._visual=None
        self._text='#'
        self._question='?'
        # Just to remove assertion error. #
        self.action_space = spaces.Discrete(6) 
        self.observation_space=spaces.Dict({
            "text": spaces.Text(min_length=1,max_length=100),"question":spaces.Text(min_length=1,max_length=100),
            "visual": spaces.Box(low=0, high=255, shape=(vga.WIDTH,vga.HEIGHT,3), dtype=np.uint8)
        })
############################################
    def _get_obs(self):
        return {"text": self._text,"question":self._question,"visual": self._visual}

    def _get_info(self):
        return {
            "progress": "currently feature not required."
        }
############################################
    def reset(self, set_no=-1, mx_timeSteps=50,seed=None, options=None):
        super().reset(seed=seed)
        self.mx_timeSteps,self.curr_time=mx_timeSteps,0
        if set_no==-1:
            self.no=np.random.randint(1,1000)
        else:
            self.no=set_no
        ## Gen initial info ##
        self.carry=False
        self.blocksLeft=[ (self.no//10**i)%10 for i in range(3) ]
        self.blocksLeft.reverse()
        self.boxType=BOXTYPE.NONE
        self._visual=vga.draw_main(self.metadata['render_modes'][self.mode],self.metadata['render_fps'],self.no)
        self.instructions,self.exp_actions = RlNlpWorld.getNLP(self.no)
        self.nlp_index = 0
        self._text = self.instructions[self.nlp_index]
        self.nlp_index = self.nlp_index+1
        return self._get_obs()
############################################
    @staticmethod
    def getNLP(no):
        global spell
        instructions = []
        exp_actions = []
        def get_word_name(no):
            digit10 = ["", "Twenty",  "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" ]
            n20 = [  "",       "One",       "Two",      "Three",
                    "Four",    "Five",      "Six",      "Seven",
                    "Eight",   "Nine",      "Ten",      "Eleven",
                    "Twelve",  "Thirteen",  "Fourteen", "Fifteen",
                    "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
            def split_no(no):
                hun,ten,uni = no//100, (no - (no//100)*100)//10, no%10
                return hun,ten,uni
            hun, ten, uni = split_no(no)
            _name = ''
            if hun: 
                _name += n20[hun] + " hundred " 
            if ten:
                if ten*10<20:
                    _name += n20[ten*10+uni]
                else:
                    _name += digit10[ten-1] + " " + n20[uni]
            else:
                _name += n20[uni]
            return _name.lower(),hun,ten,uni    
        word_name,hun,ten,uni = get_word_name(no)
        if hun:
            exp_actions = exp_actions + [0,3]*hun
            instructions = instructions  + [f'Next , pick up the {spell[_//2+1]} hundredth block .' if _%2 ==0  else 'Put the hundredth block in the hundredth\'s palce .' for _ in range(2*hun)]
        if ten:
            exp_actions = exp_actions + [1,4]*ten
            instructions = instructions  + [f'Next , pick up the {spell[_//2+1]} tenth block .' if _%2 ==0  else 'Put the tenth block in the tenth\'s palce .' for _ in range(2*ten)]
        if ten:
            exp_actions = exp_actions + [2,5]*uni    
            instructions = instructions  +  [f'Next , pick up the {spell[_//2+1]} unit block .' if _%2 ==0  else 'Put the unit block in the unit\'s palce .' for _ in range(2*uni)]       
        instructions[0] = f"This is {word_name}. Let's use our blocks to build the number. To build {word_name}"+instructions[0][len("Next"):]
        return instructions, exp_actions
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

        '''Extra reward for following instructions'''
        # reward= 1 if action in self.exp_actions else reward
        vga.carry_indicator=self.carry
        self._visual=vga.drawAgain()
        if action == self.exp_actions[self.nlp_index]:
            self.nlp_index += 1
        if self.nlp_index<len(self.instructions):
            self._text = self.instructions[self.nlp_index]
        self.curr_time += 1
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

