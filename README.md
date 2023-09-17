# Only the PPO code works in this branch.

## How to run :-
> Currently docker build is not working from inside the VMs. It's better to pull the docker image I have created. **sudo docker pull tirthankar95/rl-nlp:latest**

> First train the model using command line, an example will be 
**python3 -W ignore ppo.py --model 1 --ease -1 --log 20 --instr_type 1 --iter 500 &> m1e1.txt &**
There are two models 0 & 1; 0 -> CNN  whereas 1 -> CNN_NLP
There are four levels of dataset -1, 0, 1, 2; -1 is used for sanity check,
0 is easy dataset, 1 is medium and 2 is hard. '--log' represents how much 
detail you want while logging, default is **30(WARNING)** during training.
**iter** dictates the max. number of episodes that's being run for a dataset,
generally it's not used, unless it's a naive model used for sanity check.

> Each number in a particular dataset, let's say "easy" is given a particular number of episodes to learn it's called the time_to_learn ~ if episodeNo%time_to_learn == 0:...; The total number of episodes for a dataset is given by ~ max_episodes = max(time_to_learn*len(train_set),args.iter). ppo.py which is invoked during RL training saves the trained model with appropriate name, the variable part of the name is constructed using **suffix = [['easy','medium','hard','naive'],['fnlp_easy','fnlp_medium','fnlp_hard','fnlp_naive']]** This should match **test_path.json**, which is a pain to ensure. After the agent finishes training in the RL environment, ppo_run_nointeract.py is run. **ppo_run_nointeract.py saves (train_dict.json) and (test_dict.json) containing the results from train and test dataset in the output folder.**

> **Additional notes** - There is only the train set and test set. I am not using any sort of validation set for hyper-parameter selection. **train_result_plot** which contains data corresponding to improvement in performance is measured on the hardest example from the training set 5 times and averaged(this was done to remove randomness). 
To create multiple training seeds, I ran the code one by one and saved the results in appropriate seed folder.  

> **Running ppo_run_nointeract.py** - Before running the command below it's important to move all the elements in the local results folder ( after one RL run ) to the path described in the test_path.json file. Currently this is done manually. 
**python3 -W ignore ppo_run_nointeract.py --instr_type 0 --full_test 1 --model 1 --ease 0**
> All available valid models are run in this case. 'models_to_test' is the list of current models that can be run, if the instr_type is 1 then only the valid models which were trained using state instructions will be run. The CNN only model is present in console_output_instr0.txt

> In conclusion although ppo.py ( which is run during the training of RL model ) has to be run for individual models, ease level and instruction type. ppo_run_nointeract.py is run separately for different instruction type. 

> Folder structure ppo-run > seed#N > naive/easy/med/hard > nlp/cnn > creward.png/m1e0.txt/model/... 

## File descriptions :-

**1. ppo.py - contains the code that runs the ppo agent.**

**2. ppo_run_nointeract.py - is used to generate test results and graphs without jupyter-notebook. There is a json file test_path.json which it accesses. Unlike ppo.py where the default logging level is warning here the default logging level is info(20).**

**3. setup.py - (Ignore) is part of open ai gym framework.** 

**4. installDocker.sh - for installing docker in google cloud VMs.**

**5. run_wo_vscode.py - Using ipynb files you need to reset everytime the Env is changed. This allows to test any changes in the Env very quickly from the terminal.**

**6. model_nlp.py - contains the neural network architecture of the CNN plus NLP.**

**7. model.py - only contains the CNN architecture.**

**8. ppo_run.ipynb - to run models in an interactive way, useful for generating and visualizing graphs.**

**9. create_nlp_instructions.py - creates different types of instructions for our model. Currently there are two types of instruction that's being created. The first one is "policy" which directly feeds the policy and the second one is "state" which describes the state to the agent. For "policy" all instrctions and expected actions are created at the beginning, whereas for "state" all expected actions are created**

**10. test_path.json -**
- model_name 
    - type - 0 is CNN, 1 is CNN_NLP, it's full nlp.
    - path - is the path where the model will be save.
- train_set / test_set - contains the path where training and test sets are kept.
- train_result_plot - contains training points collected during RL training phase.
- train_result_action_reward - 
- output_path - contains creward.png, train_dict.json and test_dict.json file. It also contains the training and the test dataset along with the saved model and console output saved with the name 'm1e1.txt' (model1, easelvl1). Additionally, train_dict.json contains for each number three stats, cumulative reward, reward & action sequence used to solve the task. 


    
## Folder descriptions :-

**1. gym-examples -**

**2. gym_examples.egg-info -** 

**3. ENV_UTIL - Contains docker related files, seed information file, additional helper files.**

## Current Run Instructions :-

RUN ~ **python3 -W ignore ppo.py --model 1 --ease 0 --instr_type 1 &> m1e1i1.txt &**
Add to **test_path.json**.
	"model_fnlp_easy_stateInstr":{
		"model": {
			"type": 1,
			"path": "results/easy/nlp_stateInstr/model_fnlp_easy_stateInstr.ml"
		},
		"train_set": "results/easy/nlp_stateInstr/train_setfnlp_easy.json", 
		"test_set": "results/easy/nlp_stateInstr/test_setfnlp_easy.json", 
		"train_result_plot": "results/easy/nlp_stateInstr/test_reward_list_fnlp_easy.json",
		"output_path": "results/easy/nlp_stateInstr"
	}
Inside results create **nlp_stateInstr**.
From directory NLP_RL_Docker_Version ~ **gsutil cp -r results/*  gs://ppo-run/seed0/easy/**.

**python3 -W ignore ppo_run_nointeract.py --instr_type 0 --full_test 1 --model 1 --ease 0 &**