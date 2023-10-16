mkdir run$1
cp ~/NLP_RL_Docker_Version/results/easy/nlp_stateInstr/* run$1
cp ~/NLP_RL_Docker_Version/results/easy/nlp/* run$1
rm -rf ~/NLP_RL_Docker_Version/results/easy/nlp_stateInstr/*
rm -rf ~/NLP_RL_Docker_Version/results/easy/nlp/*
rm -rf model* train* test*
mv ~/NLP_RL_Docker_Version/console* ~/NLP_RL_Docker_Version/results/run$1
mv ~/NLP_RL_Docker_Version/m1* ~/NLP_RL_Docker_Version/results/run$1
cp ~/NLP_RL_Docker_Version/test_config* ~/NLP_RL_Docker_Version/results/run$1
cp ~/NLP_RL_Docker_Version/train_config* ~/NLP_RL_Docker_Version/results/run$1
