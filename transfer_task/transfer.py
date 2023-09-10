import torch.nn as nn 
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import torch 
import numpy as np 
import sys
sys.path.append('../')
import model_nlp as MNLP


numbers = [ (i,j) for i in range (1000)for j in range(1000) ]
def create_instructions(n_pair):
    def spell(no):
        digit10 = ["", "Twenty",  "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" ]
        n20 = ["",       "One",       "Two",      "Three",
               "Four",    "Five",      "Six",      "Seven",
               "Eight",   "Nine",      "Ten",      "Eleven",
               "Twelve",  "Thirteen",  "Fourteen", "Fifteen",
               "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        hundred, ten, unit = no//100, (no%100)//10, no%10
        _name = ''
        if hundred: 
            _name += n20[hundred] + " hundred " 
        if ten:
            if ten*10<20:
                _name += n20[ten*10+unit]
            else:
                _name += digit10[ten-1] + " " + n20[unit]
        else:
            _name += n20[unit]
        return _name.lower()
    spell_no0, spell_no1 = spell(n_pair[0]), spell(n_pair[1])
    return f'Which of the two numbers are bigger {spell_no0} or {spell_no1}?'


def create_tr_test_set():
    X, Y = [], []
    for n_pair in numbers:
        X.append(create_instructions(n_pair)) 
        if n_pair[0] > n_pair[1]: Y.append([1, 0, 0])
        elif n_pair[0] < n_pair[1]: Y.append([1, 0, 0])
        else: Y.append([0, 1, 0])
    indx = np.arange(len(X))
    tr_test_split = int(0.9 * len(X))
    Y = Y[indx]
    Ytr, Ytest = Y[:tr_test_split], Y[tr_test_split:]
    X_ = []
    for ids in indx:
        X_.append(X[ids])
    Xtr, Xtest = X[:tr_test_split], X[tr_test_split:]
    return Xtr, Ytr, Xtest, Ytest


if __name__ == '__main__':
    # Validation function
    def validate(model, Xval, Yval):
        model.eval()
        with torch.no_grad():
            lb, r_loss = 0, 0
            val_batch_size = 256
            criterion = nn.CrossEntropyLoss() 
            while lb + batch_size < Yval.shape[0]:
                Yp_val = model(Xval[lb: lb+val_batch_size])
                r_loss += criterion(Yp_val, Yval[lb: lb+val_batch_size]).item()
                lb += val_batch_size
            if lb < Yval.shape[0]:
                Yp_val = model(Xval[lb:])
                r_loss += criterion(Yp_val, Yval[lb:]).item()
            return r_loss / Yval.shape[0]

    model = MNLP.NNModelNLP()
    '''
    LOAD MODELS HERE FOR TRANSFER TASK.
    '''
    Xtr, Ytr, Xtest, Ytest = create_tr_test_set()
    tr_val_split = int(0.9 * Xtr.shape[0])
    Xtr, Xval = Xtr[:tr_val_split], Xtr[tr_val_split:]
    Ytr, Yval = Ytr[:tr_val_split], Ytr[tr_val_split:]
    epochs = 20
    batch_size = 128 
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    tr_loss, val_loss = [], []
    for epoch in range(epochs):
        lb, run_tr_loss = 0, 0
        while lb + batch_size <= Ytr.shape[0]:
            Yp = model(Xtr[lb: lb+batch_size])
            optimizer.zero_grad()
            loss = criterion(Yp, Ytr[lb: lb+batch_size])
            loss.backward()
            with torch.no_grad(): run_tr_loss += loss.item()
            optimizer.step()
            lb += batch_size
        if lb < Ytr.shape[0]:
            Yp = model(Xtr[lb:])
            optimizer.zero_grad()
            loss = criterion(Yp, Ytr[lb:])
            loss.backward()
            optimizer.step()
            with torch.no_grad(): run_tr_loss += loss.item()
        tr_loss.append(run_tr_loss)
        val_loss.append(validate(model, Xval, Yval))
        if len(val_loss) >= 2 and val_loss[-1] >= val_loss[-2]:
            break
    # TEST MODEL.
    Yp_test = model(Xtest)
    acc = (torch.argmax(Yp_test, dim = 1) == torch.argmax(Ytest, dim = 1)).sum().item()
    print(f' TEST ACCURACY [{acc*100/Yp_test}]%')


