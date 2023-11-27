class Data:
    def __init__(self, state, stateNLP, action, log_prob, returns, advantage, completed):
        self.state = state 
        self.stateNLP = stateNLP 
        self.action = action
        self.log_prob = log_prob
        self.returns = returns 
        self.advantage = advantage
        self.len = state.shape[0]
        self.completed = completed

class BestData:
    def __init__(self) -> None:
        self.dict = {}
    def setter(self, no, state, stateNLP, action, log_prob, returns, advantage):
        obj = Data(state, stateNLP, action, log_prob, returns, advantage)
        if not obj.completed: return 
        if no not in self.dict:
            self.dict[no] = obj 
        elif self.dict[no].len > obj.len:
            self.dict[no] = obj
    def getter(self, no):
        if no not in self.dict:
            return None, None, None, None, None, None
        return self.dict[no].state, self.dict[no].stateNLP, \
               self.dict[no].action, self.dict[no].log_prob, \
               self.dict[no].returns, self.dict[no].advantage