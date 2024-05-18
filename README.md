**How to run the code?**
0. Git clone the project 'git@github.com:tirthankarCU/NumberLearningInChildren_ML.git' and go inside one directory.
1. wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove*.zip
2. First pull the environment for running the code from docker-hub using. 
   docker pull tirthankar95/rl-nlp:latest
3. sudo docker run -p 8888:8888 -v $(pwd):/NumberLearningInChildren_ML tirthankar95/rl-nlp &
4. sudo docker ps 
   -> Get the name of the image.
5. sudo docker exec -it angry_lewin bash 
   -> Here "angry_lewin" is the name of the container
6. python3 -W ignore RL_Algorithm/PPO/ppo.py &> Results/console.log &
7. python3 -W ignore RL_Algorithm/PPO/ppo_post_run.py &