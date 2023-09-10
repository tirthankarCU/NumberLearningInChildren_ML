# Only the PPO code works in this branch.

## File descriptions :-

**1. ppo.py - contains the code that runs the ppo agent.**
**2. ppo_run_nointeract.py - is used to generate test results and graphs without jupyter-notebook. There is a json file which**
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
- train_set / test_set - contains the path where training and test sets are kept
- train_result_plot - contains training points collected during training phase
- output_path - contains creward.png, train_dict.json and test_dict.json file.In train_dict.json for each number 3 things are stored, first is cumulative reward, second is action_arr and then is the reward array. 

    
## Folder descriptions :-

**1. gym-examples -**
**2. gym_examples.egg-info -** 