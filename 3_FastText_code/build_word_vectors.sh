./fasttext skipgram -input data/agnews_train_unlabeled.txt -output data/agnews.model -wordNgrams 2 -epoch 15 -verbose 2 -dim 100
./fasttext skipgram  -input data/tripadvisor_train_unlabeled.txt -output data/tripadvisor.model -wordNgrams 2 -epoch 15 -verbose 2 -dim 100
./fasttext skipgram  -input data/amazon_six_train_unlabeled.txt -output data/amazon_six.model -wordNgrams 2 -epoch 15 -verbose 2 -dim 100
./fasttext skipgram  -input data/dbpedia_train_unlabeled.txt -output data/dbpedia.model -wordNgrams 2 -epoch 15 -verbose 2 -dim 100
./fasttext skipgram  -input data/yahoo_a_train_unlabeled.txt -output data/yahoo.model -wordNgrams 2 -epoch 15 -verbose 2 -dim 300
./fasttext skipgram  -input data/yelp_f_train_unlabeled.txt -output data/yelp.model -wordNgrams 2 -epoch 15 -verbose 2 -dim 300
