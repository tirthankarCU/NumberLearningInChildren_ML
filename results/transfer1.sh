mkdir run$1

cp naive/nlp_stateInstr/* run$1
cp naive/nlp/* run$1
rm -rf naive/nlp_stateInstr/*
rm -rf naive/nlp/*

cp easy/nlp_stateInstr/* run$1
cp easy/nlp/* run$1
rm -rf easy/nlp_stateInstr/*
rm -rf easy/nlp/*

rm -rf model* train* test*
mv ../console* run$1
mv ../m1* run$1
cp ../test_config* run$1
cp ../train_config* run$1
