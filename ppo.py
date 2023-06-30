import argparse


'''
    model_0: Naive CNN
    model_1: Full NLP
    model_2: Partial NLP
'''

'''
    easy_0
    medium_1
    difficult_2
'''
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NLP_RL parameters.')
    parser.add_argument('--model',type=int,help='Type of the model.')
    parser.add_argument('--ease',type=int,help='Level of ease you want to train.')
    args=parser.parse_args()
