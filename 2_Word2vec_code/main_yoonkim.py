import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D,Dropout
# from keras.models import Sequential # For RNN-LSTMs

import numpy as np
from numpy import zeros

import json, os, sys, argparse

parser = argparse.ArgumentParser(description='input arguments')

parser.add_argument('--log_file_name', type=str, default="", help='name of file you want to save')
parser.add_argument('--output_path', type=str, default="./logs/", help='where is the output path?')

parser.add_argument('--max_seq', type=int, default='300', help='max sequence')
parser.add_argument('--embed_dim', type=int, default='100', help='embedding size')
parser.add_argument('--nhid', type=int, default=100, help='number of classifier hidden units')
parser.add_argument('--training_hidden_state', action="store_true", help='use training_hidden_state')
# parser.add_argument('--test_instance', type=int, default=0, help='number of test instance')
parser.add_argument('--val_percent', type=float, default=0.2, help='validation percent in train set')
# parser.add_argument('--cross_val', type=int, default=0, help='Cross validation, how many folds?')

parser.add_argument('--seed', type=int,  default=2212, help='set seed')
parser.add_argument('--epoch', type=int,  default=15, help='specify epoch')
parser.add_argument('--early_stop', type=int,  default=4, help='early stop patience')

args = parser.parse_args()


from tensorflow import set_random_seed
set_random_seed(args.seed)
np.random.seed(args.seed)



def get_tag_and_training_data(filename, label_dict=False):
    '''
    takes the input file and returns  tokenized sentences and document tags as separate lists
    How to use:
    Y, X, label_list = get_tag_and_training_data(".../train.txt")
    '''
    original_labels = list()
    initial_label_dict   = {"label2index":{}, "index2label":{}}
    input_label = label_dict
    texts, labels = list(), list()

    with open(filename, encoding='utf8') as f:
        for line in f:
            #Initialize the token list for line
            words  = line.split()
            label = words[0]

            # Original labels
            original_labels.append(label)

            if not input_label:
                label_dict = initial_label_dict
                if label in label_dict["label2index"]: pass
                else :
                    index = len(label_dict["label2index"])
                    label_dict["label2index"][label]= index
                    label_dict["index2label"][index] = label
                    print(index, label, label_dict)

            label_idx = label_dict["label2index"][label]
            labels.append(label_idx)
            sent  = ' '.join(words[1:])
            texts.append(sent)

    return labels, texts, label_dict





#  ---------------------- ---------- -------------------
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static
#  ---------------------- ---------- -------------------


# Model Hyperparameters
embedding_dim = EMBEDDING_DIM = args.embed_dim
filter_sizes = (3, 8)
num_filters = 10

# Original hyperparameters
# filter_sizes = (3, 4, 5)
# num_filters = 100

dropout_prob = (0.5, 0.8)
hidden_dims = args.nhid
num_epochs = args.epoch
VALIDATION_SPLIT = args.val_percent
MAX_NUM_WORDS = 10000

# Prepossessing parameters
sequence_length = MAX_SEQUENCE_LENGTH = args.max_seq # number of words in one sentence. Let's say 500 is maximum.

# Word2Vec hyperparameters (see train_word2vec)
min_word_count = 3
context = 7



# ---------------------- Hyperparameters setting for each dataset  -----------------------

target_list     = ["agnews", "dbpedia", "tripadvisor", "amazon_six", "yahoo_a", "yelp_f"]

emb_size_list   = [100, 100, 100, 100, 100, 100]

instance_list   = [10, 50, 100, 200, 500, 1000, 2000, ""]
batch_size_list = [2,  2,   2,   2,   20,   20,   20, 256]




# ------------------------ Start training -----------------------
for target, emb_size in zip(target_list, emb_size_list):
    for instance, batch_size in zip(instance_list, batch_size_list):
        embedding_dim = EMBEDDING_DIM = emb_size

        # ============= Load files and convert texts to seqeuence ==========
        train_file_path = "/TODO_TYPE_A_DIRECTORY_OF_SIX_DATASETS/Target_{}/{}_train{}.txt".format(target, target, instance)
        test_file_path  = "/TODO_TYPE_A_DIRECTORY_OF_SIX_DATASETS/Target_{}/{}_test.txt".format(target, target)

        y_train, X_train, label_dict = get_tag_and_training_data(train_file_path, label_dict=False)
        y_test,  X_test,  label_dict = get_tag_and_training_data(test_file_path, label_dict=label_dict)


        # 2. Vectorize text: fit on texts
        tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)             # seq <- texts
        X_test_seq  = tokenizer.texts_to_sequences(X_test)

        word_index = tokenizer.word_index
        print(' unique token number : %s ' % len(word_index))

        vocabulary_inv = dict((v, k) for k, v in word_index.items())
        vocabulary_inv[0] = "<PAD/>"

        X_train = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH) # data <- seq
        y_train = to_categorical(np.asarray(y_train))

        X_test = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH) # data <- seq
        y_test = to_categorical(np.asarray(y_test))

        x_train = np.array(X_train)
        y_train = np.array(y_train)
        x_test = np.array(X_test)
        y_test = np.array(y_test)

        print("Word index : \n",len(word_index))


        if sequence_length != x_test.shape[1]:
            print("Adjusting sequence length for actual size")
            sequence_length = x_test.shape[1]

        print("x_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)
        print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

        # Prepare embedding layer weights and convert inputs for static model
        print("Model type is", model_type)
        if model_type in ["CNN-non-static", "CNN-static"]:
            embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, file_head=target, num_features=embedding_dim, min_word_count=min_word_count, context=context)

            if model_type == "CNN-static":
                x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
                x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
                print("x_train static shape:", x_train.shape)
                print("x_test static shape:", x_test.shape)

        elif model_type == "CNN-rand":
            embedding_weights = None
        else:
            raise ValueError("Unknown model type")




        # ========================== Build model ==========================
        if model_type == "CNN-static":
            input_shape = (sequence_length, embedding_dim)
        else:
            input_shape = (sequence_length,)

        model_input = Input(shape=input_shape)

        # Static model does not have embedding layer
        if model_type == "CNN-static":
            z = model_input
        else:
            z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

        z = Dropout(dropout_prob[0])(z)

        # Convolutional block
        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob[1])(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(len(label_dict["label2index"]), activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        # ========================== Build model ==========================




        # ================ Initialize embedding layer with pretrained-word2vec ================
        if model_type == "CNN-non-static":
            weights = np.array([v for v in embedding_weights.values()])
            print("Initializing embedding layer with word2vec weights, shape", weights.shape)
            embedding_layer = model.get_layer("embedding")
            embedding_layer.set_weights([weights])




        # ================================  Train the model  ================================
        from keras.callbacks import ModelCheckpoint
        from keras.callbacks import EarlyStopping

        filepath="./checkpoints/12Aril_{}.hdf5".format(target)
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        callbacks_list = [
            EarlyStopping(monitor='val_loss', patience=args.early_stop, verbose=2),
            ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        ]

        # class_weight = {0: 0, 1: 1, 2:50}
        # history = model.fit(x_train, y_train, batch_size=256, epochs=50, class_weight = class_weight, callbacks = callbacks_list, validation_split=0.2, verbose=1)

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=args.epoch, callbacks = callbacks_list, validation_split=args.val_percent, verbose=1)



        # ================================  Load best model  ================================
        from keras.models import load_model
        model = load_model("./checkpoints/{}.hdf5".format(target))



        # ================================  Prediction  ================================
        y_prob = model.predict(x_test)

        y_pred = [label_dict["index2label"][int(np.where(prob==max(prob))[0][0])] for prob in y_prob]
        y_true = [label_dict["index2label"][int(np.where(v==1.0)[0])] for v in y_test]


        # ================================  REPORT  ================================
        from smnlp_general import report_matrix
        description = "\n instance per class {}, epoch: {}, optimizer : {}\n".format(instance if instance != "" else "total", args.epoch, "adam")
        title = str(target)+'.report'
        # description += "Sample of Y_test : {}, \n Y_pred : {}\n".format(y_true[:10], y_pred[:10])
        report_matrix("./report/", description, title, y_true, y_pred, append=True, torch=False)

                
