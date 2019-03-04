### Word2vec model is based on "https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras.git"
All pre-trained word vectors are pre-trained by the above codes. <br>


Run the 'main_yoonkim.py' code.<br>
$ python main_yoonki.py --log_file_name "log_files" --max_seq 300 --embed_dim 300 --nhid 100<br>

parser.add_argument('--training_hidden_state', action="store_true", help='use training_hidden_state')<br>
parser.add_argument('--val_percent', type=float, default=0.1, help='validation percent in train set')<br>
parser.add_argument('--seed', type=int,  default=2212, help='set seed')<br>
parser.add_argument('--epoch', type=int,  default=15, help='specify epoch')<br>
parser.add_argument('--early_stop', type=int,  default=4, help='early stop patience')<br>
