import copy
import torch 
import pytest 
import logging 

class Data:
    def __init__(self, state, stateNLP, action, log_prob, returns, advantage):
        self.state = copy.deepcopy(state) 
        self.stateNLP = copy.deepcopy(stateNLP) 
        self.action = copy.deepcopy(action)
        self.log_prob = copy.deepcopy(log_prob)
        self.returns = copy.deepcopy(returns) 
        self.advantage = copy.deepcopy(advantage)
        self.len = action.shape[0]

class BestData:
    def __init__(self) -> None:
        self.dict = {}
    def setter(self, no, state, stateNLP, action, log_prob, returns, advantage, completed):
        if not completed or \
          (no in self.dict and self.dict[no].len <= action.shape[0]): 
            return
        self.dict[no] = Data(state, stateNLP, action, log_prob, returns, advantage) 

    def getter(self, no):
        if no not in self.dict:
            return None, None, None, None, None, None
        return self.dict[no].state, self.dict[no].stateNLP, \
               self.dict[no].action, self.dict[no].log_prob, \
               self.dict[no].returns, self.dict[no].advantage

def test_dict_sanity():
    obj_bd = BestData()
    tr_ex = 10
    obj_bd.setter(10, torch.randn(size = (tr_ex, 5, 5)), None, \
                  torch.randn(size = (tr_ex, 5, 5)), None, None, None, True)
    for k, v in obj_bd.dict.items():
        logging.info(f'No[{k}] ~ Len[{v.len}]')
    assert True 