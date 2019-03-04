import torch
import pickle
from torch.autograd import Variable

def DEPRECATED_get_word_dict(file_path):
    """
    How to use :
    file_path = "awd_word_dictionary.obj"
    word_dict = get_word_dict(file_path)
    Load word vocabulary dictionary
    Need to import data.py in same directory
    """
    import pickle
    import inspect

    word_dict = None
    with open(file_path, 'rb') as f:
    # with open("word_dictionary.obj", 'rb') as f:
        word_dict = pickle.load(f)

    if inspect.isclass(type(word_dict)):
        print("word dictionary keys : ", type(word_dict), len(word_dict))
        print("index for <eos> is : ", word_dict.word2idx['<eos>'])
    elif isinstance(word_dict, dict):
        print("word dictionary keys : ", type(word_dict), word_dict.keys())
        print("index for <eos> is : ", word_dict['word2index']['<eos>'])

    return word_dict


def get_train_loader(file_path, batch_size=10, shuffle=True, num_workers=4):
    """
    How to use :
    file_path = "/Users/sungmin/Desktop/transfer_learning/Dataset/Seal-Software/41k_downsampled/2k_sample_provisions.txt"
    trainloader = get_train_loader(file_path)
    for
    """
    from torch.utils.data import DataLoader

    traindata = list()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            traindata.append(line.rstrip('\n'))
    # Define DataLoader, use it for every batch iteration
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return trainloader


def text2idxs(X_data, word_dict, verbose=False):
    """
    How to use : text2idxs(X_test, word_dict)

    Change text to index
    e.g.,
    Original text :
    ['this is how it works . <eos>',
    'it was snow . <eos>',
    'cat sat on the mat . <eos>',
    'people cry for nothing . <eos>']

    Converted to idx:
    [ [479, 26, 1015, 46, 480, 15, 0],
        [46, 131, 12284, 15, 0],
        [5887, 4777, 93, 17, 27025, 15, 0],
        [1239, 10712, 39, 7142, 15, 0]   ]
    """
    X_text_idx = []
    unk_cnt, voca_cnt = 0, 0
    for line in X_data:
        temp = []
        for word in line.split():
            try : temp.append(word_dict.word2idx[word])
            except :
                temp.append(word_dict.word2idx["<unk>"])
                unk_cnt +=1
                # rint("Cannot find a word in dictionary")
            finally:
                voca_cnt +=1
        X_text_idx.append(temp)

    if verbose :
        print("Total unknown word count is ", unk_cnt)
        print("Total vocabulary count is ", voca_cnt)

    return X_text_idx



def idx2Var_Getseqlen(text_indxs, label_indxs, pad_number=0, DEBUG=False):
    """
    Updated on 28th, Feb, 2018.
    It returns reordered text_indxs, labels_indxs
    How to use :
    it is recommened to use with next Variable as a pair.
    => inputs_var_tensor, seq_lengths = idx2Var_Getseqlen(X_test_indx)
    => labels_var_tensor = torch.autograd.Variable(torch.Tensor(y_test))

    """
    seq_length_label = [(len(text_indxs[i]), label) for i, label in enumerate(label_indxs)]
    reordered_labels = sorted(seq_length_label, key=lambda x: x[0], reverse=True)
    reordered_labels = [label for len, label in reordered_labels]

    if DEBUG : print('\n\n label_indxs is ', label_indxs, '\n\n')
    if DEBUG : print('\n\n seq_length_label is ', seq_length_label, '\n\n')
    if DEBUG : print('\n\n reordered_labels is ', reordered_labels, '\n\n')


    seq_len = [len(x) for x in text_indxs]

    # sort by length for pytorch(support decreasing shape list)
    text_indxs.sort(key=lambda x: len(x), reverse=True)
    seq_len.sort(reverse=True)

    MAX_LEN = seq_len[0]

    if DEBUG : print('\n\n seq_len is ', seq_len, '\n\n')

    # Padding
    for indx in text_indxs:
        while len(indx) < MAX_LEN:
            indx += [pad_number]

    # # Quick padding
    # def pad(text_seq, length):
    #     tensor = torch.LongTensor(text_seq)
    #     return torch.cat([tensor, tensor.new(length-tensor.size(0), *tensor.size()[1:]).zero_()])

    labels_var_tensor = torch.autograd.Variable(torch.LongTensor(reordered_labels))
    inputs_var_tensor = torch.autograd.Variable(torch.LongTensor(text_indxs))

    # return torch.autograd.Variable(torch.LongTensor(text_indxs)), seq_len
    return inputs_var_tensor, labels_var_tensor, seq_len



# batct size is ~10~12 by default
def batchify_4_eachline(text_data, word_dict, batch_size=10): # @TODO: Use arg later., shuffle=True
    """
    How to use :
    batchified_texts, batchified_labels, label_dict = batchify_4_eachline(text_data, word_dict, batch_size=10)

    Input : one line with label + text(not index)
    Input : CLASS1 this is sample line of annotated data .

    Output : bathchified text_idx_data, represented as index in dictionary
    Output : e.g., line1 : CLASS1 232 5512 591 2 443  89 51 <-- not padded

    batch_size is equal to bsz(number of sentences in one batch) : it means number of lines(sentences).
    Make a batch for annoated text file.
    This batchify function only covers one line for one class
    """
    from utils_sm import text2idxs
    import numpy as np
    # from copy import deepcopy

#     nbatch = len(text_data) // batch_size

    batchfied_text_idxs = list()
    batchfied_labels    = list()

    previous_flag = 0

    for _ in range(0, len(text_data), batch_size):
#         print(line, len(text_data), batch_size)

        one_batch_label, one_batch_text = list(), list()
        lines = text_data[previous_flag:previous_flag+batch_size]
        for line in lines:
            words = line.split()
            label = words[0]
            text = ' '.join(words[1:])

            one_batch_label.append(label)
            one_batch_text.append(text)

#         print("*"*50, one_batch_text)

        text2idx_converted = text2idxs(one_batch_text, word_dict)

        batchfied_labels.append(one_batch_label)
        batchfied_text_idxs.append(text2idx_converted)

        previous_flag += batch_size

    # # Every epoch, make it shuffle before feeding to batch iteration.
    # if shuffle==True:
    #     random_idx = np.random.permutation(len(batchfied_text_idxs))
    #     print("random idx : ", random_idx[:100])
    #     batchfied_text_idxs_, batchfied_labels_ = deepcopy(batchfied_text_idxs), deepcopy(batchfied_text_idxs)
    #     for i, j in enumerate(random_idx):
    #         batchfied_text_idxs[i] = batchfied_text_idxs_[j]
    #         batchfied_labels[i]    = batchfied_labels_[j]
    #
    #     del batchfied_text_idxs_, batchfied_labels_
    #
    # print(type(batchfied_labels), batchfied_labels[:2], batchfied_labels[0][0])
    label_list = list(set([a for b in batchfied_labels for a in b])) # flatten -> set
    label2idx  = {label: i for i, label in enumerate(label_list)}
    idx2label  = {i: label for i, label in enumerate(label_list)}

    label_dict = {"label2idx" : label2idx, "idx2label" : idx2label}

    return batchfied_text_idxs, batchfied_labels, label_dict



def get_data_target(batchified_text, batchified_label, i, label_dict, evaluation=False, pad_number=0): # @TODO: use arg later. + when to use evaluation?
    """
    How to use :
    inputs_var_tensor, labels_var_tensor, seq_lengths = get_data_target(batchified_texts, batchified_labels, i, label_dict)

    data is a total file of annotated data.
    batct size is ~10~12 for text classification
    batch size is 10 by default in this code.

    e.g.,
    line1 : CLASS1 this is sample of annotated data.
    line2 : CLASS2 this is class2.
    ...
    """
    from utils_sm import idx2Var_Getseqlen

    if i== -1 :
        one_batch_sents = batchified_text
        one_batch_label = batchified_label
    else:
        one_batch_sents = batchified_text[i] # [[121, 2322, 493, 51], [512, 331, 612, 10], ...]
        one_batch_label = batchified_label[i] # ['ChangeOfControl', 'ForceMajeure', 'ForceMajeure'...]

    one_batch_label = [label_dict['label2idx'][label] for label in one_batch_label] # [0, 1, 1, ...]

    inputs_var_tensor, seq_lengths = idx2Var_Getseqlen(one_batch_sents, pad_number=pad_number)
    # Reshape inputs_var_tensor
    # inputs_var_tensor = inputs_var_tensor.view(inputs_var_tensor.size(0), inputs_var_tensor.size(1))

    labels_var_tensor = torch.autograd.Variable(torch.Tensor(one_batch_label))
    labels_var_tensor = labels_var_tensor.view(labels_var_tensor.size(0), -1)
    # print(labels_var_tensor, labels_var_tensor.size())

    return inputs_var_tensor, labels_var_tensor, seq_lengths


def run_sample():

    FILE_PATH = "/Users/sungmin/Desktop/transfer_learning/Dataset/Seal-Software/41k_downsampled/41k_downsampled.txt"
    text_data = list()
    with open(FILE_PATH, 'r', encoding='utf8') as f:
        for line in f:
            text_data.append(line)

    batchified_texts, batchified_labels, label_dict = batchify_4_eachline(text_data, word_dict, batch_size=10)

    i = 0
    batch_one = get_data_target(batchified_texts, batchified_labels, i, label_dict)
    i +=1
    batch_two = get_data_target(batchified_texts, batchified_labels, i, label_dict)


def get_train_loader(file_path, batch_size=10, shuffle=True, num_workers=4):
    """
    How to use :
    file_path = "/Users/sungmin/Desktop/transfer_learning/Dataset/Seal-Software/41k_downsampled/2k_sample_provisions.txt"
    trainloader = get_train_loader(file_path)
    for
    """
    from torch.utils.data import DataLoader

    traindata = list()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            traindata.append(line.rstrip('\n'))
    # Define DataLoader, use it for every batch iteration
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return trainloader

def get_word_dict(file_path, check_word2index=None, check_index2word=None):
    """
    How to use :
    file_path = "awd_word_dictionary.obj"
    word_dict = get_word_dict(file_path)
    Load word vocabulary dictionary
    Need to import data.py in same directory
    """
    import pickle
    import inspect

    word_dict = None
    with open(file_path, 'rb') as f:
    # with open("word_dictionary.obj", 'rb') as f:
        word_dict = pickle.load(f)

    print('--- Loading word dictionary ---', type(word_dict))
    # if inspect.isclass(type(word_dict)):
    #     print("word dictionary keys : ", type(word_dict), len(word_dict))
    #     print("index for <eos> is : ", word_dict.word2idx['<eos>'])

    # print("word dictionary keys : ", word_dict.keys())
    # print("Total voca size : ", len(word_dict['word2index']))
    # if check_word2index :
    #     print("index for {} is : {}".format(check_word2index, word_dict['word2index'][check_word2index]))
    # if check_index2word or check_index2word==0 :
    #     print("word for {} is : {}".format(check_index2word, word_dict['index2word'][check_index2word]))

    return word_dict


def get_label_dict(file_path=False, dataloader=False, label_position=0):
    """
    Input as labeled file.
    e.g.,
    one line : label, word1, word2, word3, ...
    label_dict = get
    label_dict = get_label_dict(trainloader) # after defining trainloader
    """
    from torch.utils.data import DataLoader
    
    if file_path:

        label_dict = {'label2index': {}, 'index2label': {}}
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                label = line.split()[label_position]
                index = label_dict['label2index']
                label_dict['label2index'] = index
                label_dict['index2label'] = label

        return label_dict

    elif dataloader:
        label_list = list(set([words.split()[label_position] for line in dataloader for words in line]))
        label_dict = {"label2index": {l:label_list.index(l) for l in label_list},
                    "index2label": {label_list.index(l):l for l in label_list}}
        del label_list
        return label_dict




def describe(model_or_dict):
    try :
        for name, param in model_or_dict.state_dict().items():
            print(name, param.size())
        print(model_or_dict)
    except :
        for name, param in model_or_dict.items():
            print(name, param.size())
    return None



def load_pretrained_model(model, pretrained_dict=None, pre_model_path=None):
    """
    How to use :
    model = load_pretrained_model(model, pretrained_dict, "smAWD.pt")

    one shot: model.load_state_dict(torch.load(pre_model_path))
    
    it seems that, this function occur forcing GPU use.
    """
    import torch

    if pre_model_path:
        pretrained_dict = torch.load(pre_model_path)
    elif pretrained_dict is None:
        raise ValueError("You must give one of pretrained_dict or pre_model_path")


    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pre_weights = list(model_dict.keys())
    current_weights = list(pretrained_dict.keys())
    excluded_weights = [k  for k in pre_weights if k not in current_weights]

    # 2. overwrite entries in the existing state dict
    try : print("Before update encoder weight: ", model.encoder.weight)
    except : print("Loading stroed weights")
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    # model.load_state_dict(pretrained_dict) or
    model.load_state_dict(model_dict)
    try : print("After update encoder weight: ", model.encoder.weight)
    except : pass

    print("Successfully loaded pretrained weights, except weights in : ", excluded_weights)
    del [model, model_dict, pretrained_dict, current_weights, excluded_weights]
    
    return None


# ================== Deprecated, replaced by oneshot_padding========================
def get_text_label(lines, word_dict, label_dict, eos=True, label_position=0):
# ================== Deprecated, replaced by oneshot_padding========================

    """
    How to use:
    inputs_var_tensor, labels_var_tensor, seq_lengths = get_text_label(lines, word_dict, label_dict)
    """
    from utils_sm import idx2Var_Getseqlen
    import re

    sentences_idx = []
    labels_idx    = []
    for line in lines:
        # line = re.sub(r'\b\W+\b', ' ', line)
        words = line.split()
        label = words[label_position]
        one_line = []
        words = words[1:]
        for w in words:
            try : one_line.append(word_dict.word2idx[w])
            except :
                # word_dict.word2idx["<unk>"] -> 26
                one_line.append(word_dict.word2idx["<unk>"])

        # Add end of sentence symbol.
        if eos : one_line.append(word_dict.word2idx["<eos>"]) # index -> 24
        # replace to text_idnex
        sentences_idx.append(one_line)
        labels_idx.append(label_dict["label2idx"][label])


    # Change to text to sequence, get seq_length
    inputs_var_tensor, labels_var_tensor, seq_lengths = idx2Var_Getseqlen(sentences_idx, labels_idx, DEBUG=False)
    # Replace eidx2idx2Var_Getseqlen part with function padding_reodrering()
    return inputs_var_tensor, labels_var_tensor, seq_lengths


def oneshot_padding(sentences, label_dict, word_dict, unk_symbol="<unk>", eos_symbol="<eos>", volatile=False):
    import torch

    vectorized_seqs = []
    label_list = []

    for sent in sentences:
        sent_seq = []
        words = sent.split()
        label = words[0]
        words = words[1:]

        for token in words:
            try : sent_seq.append(word_dict["word2index"][token])
            except : sent_seq.append(word_dict["word2index"][unk_symbol])

        sent_seq.append(word_dict["word2index"][eos_symbol]) # Add end of sentence symbol
        vectorized_seqs.append(sent_seq)
        label_list.append(label_dict["label2index"][label])

    # print(vectorized_seqs)
    # vectorized_seqs = [[voca_dict["word2index"][tok] for tok in seq] for seq in sentences]

    seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

    # Padding with zeros
    seq_tensor = torch.autograd.Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max())), volatile=volatile).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    # print(seq_tensor)

    # Sorting
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Rerordering labels with perm_idx
    # print("before reorder labels", label_list)
    labels_tensor = torch.LongTensor([label_list[idx] for idx in perm_idx])
    labels_tensor = torch.autograd.Variable(labels_tensor)
    # print("after reorder labels", labels_tensor)

    return seq_tensor, labels_tensor, seq_lengths



class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def play_with_layers():
# # Add layer by adding add_layer() function
# model.add_layer()

# # Replace layer in model by redefining modeuls
# # before_train_new_layer1_weight = model.new_layer1.weight.clone()
# new_hidden_in, new_hidden_out = 100, 200

# # Replace layer
# model._modules['decoder'] = nn.Linear(new_hidden_in, new_hidden_out)

# # # # Discard layer
# # import torch.nn as nn
# # # clip_model = nn.Sequential(*list(new_model.children())[:-1])    # discard last layer
# # model = nn.smModel(*list(model.children())[4:])   # discard first 3layers and last layer.
# Freeze

# for param in current_model.parameters():
#     param.requires_grad = False
#
# if torch.cuda.is_available():
#     current_model.cuda()

# fcLayers = nn.Sequential(
#     # stop at last layer
#     *list(current_model.children())[:-1]
# )
# fcLayers[0]

# describe_model(model)
    return None



def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def freeze_layers(model, indice=None, verbose=1):
    """Important fact is,
    # for name, param in model.state_dict().items() doesn't change anything
    # NOTE, USE model.parameters()
    # NOTE, USE model.parameters()
    How to use:

    """
    layer_names = []
    if verbose==1:
        for i, (name, param) in enumerate(model.state_dict().items()):
            layer_names.append(name)
            print("Total weight of layers - ({}) : {}".format(i, name))
    elif verbose==0:
        pass

    if indice is None :
#         user_input = input("Please feed indice, 0 for freezing layers 1 for active layers")
        if verbose==1: print("Make all layers trainable")
        for param in model.parameters():
            param.requires_grad = True

    else:
        # Masking with range
        for i, param in enumerate(model.parameters()):
            if i >= indice[0] and i<= indice[1]:
                param.requires_grad = False
            else: param.requires_grad = True


    # print, frozen layers
    if verbose==0: pass
    else:
        for name, param in zip(layer_names, model.parameters()):
            print("layer name : {:3}, Frozen? {:>5}".format(name, "Trainable" if param.requires_grad else "$$ Frozen $$"))
        print("Freezing layer is done.")
        
    return model

def filter_frozen_layers(model):
    """
    How to use : optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    """
    return optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)



# DEPRECATED
def DEPRECATED_eval(model, data_loader, label_dict, word_dict, batch_size, trained_hidden=None,
multi_models=False, multi_models_dict=False, model_cat=False, cuda=True, trial=1000,
unk_symbol="<unk>", eos_symbol="<eos>", seal=False):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import confusion_matrix

    trial_cnt = 0

    y_true, y_pred = list(), list()
    flag_escape_loop = False

    model.eval()


    for lines in data_loader:
        multi_seq_tensors = []
        multi_seq_lengths = []
        multi_outputs     = []

        if multi_models and multi_models_dict and model_cat:

            for multi_model, multi_word_dict in zip(multi_models, multi_models_dict):
                multi_model.eval()
                seq_tensor, labels_tensor, seq_lengths = oneshot_padding(lines, label_dict, multi_word_dict, unk_symbol=unk_symbol, eos_symbol=eos_symbol)

                if cuda:
                    seq_tensor = seq_tensor.cuda(async=True)

                seq_tensor = seq_tensor.t().contiguous()

                multi_seq_tensors.append(seq_tensor)

                multi_model.save_seq_lengths(seq_lengths)
                output, _ = multi_model(seq_tensor)
                multi_outputs.append(output)


        seq_tensor, labels_tensor, seq_lengths = oneshot_padding(lines, label_dict, word_dict, unk_symbol=unk_symbol, eos_symbol=eos_symbol)
        if  cuda :
            seq_tensor = seq_tensor.cuda(async=True)
            labels_tensor = labels_tensor.cuda(async=True)

#             hidden = model.init_hidden(batch_size)


        model.save_seq_lengths(seq_lengths)
        seq_tensor = seq_tensor.t().contiguous()
        target = labels_tensor.squeeze()

        if trained_hidden:
            result, _ = model(seq_tensor, trained_hidden)
        elif not trained_hidden:
            result, _ = model(seq_tensor)

        multi_outputs += [result]
        cat_input = torch.cat(multi_outputs, 1)
        del multi_outputs
        
        result = model_cat(cat_input)
        _, predicted = torch.max(result, 1)
        
        y_true.append(target.data)
        y_pred.append(predicted.data)


        if trial_cnt >= trial: break
        trial_cnt += batch_size

    if torch.cuda.is_available():
        y_true = torch.cat(y_true).cpu().numpy().astype(int)
        y_pred = torch.cat(y_pred).cpu().numpy().astype(int)
    else:
        y_true = torch.cat(y_true).numpy().astype(int)
        y_pred = torch.cat(y_pred).numpy().astype(int)

    if not seal:
        y_true = [label_dict["index2label"][i] for i in y_true]
        y_pred = [label_dict["index2label"][i] for i in y_pred]
    elif seal:
        y_true   = [label_dict["index2label"][i] for i in y_true]
        y_pred   = [label_dict["index2label"][i] for i in y_pred]
        y_true_refined = []
        y_pred_refined = []

        for i, pred_label in enumerate(y_pred):
            true_label = y_true[i]
            if true_label == "NONE" and pred_label == "NONE":
                # Skip the
                continue

            else:
                y_true_refined.append(y_true[i])
                y_pred_refined.append(y_pred[i])
        y_true = y_true_refined
        y_pred = y_pred_refined


    print("F-2 score : ", fbeta_score(y_true, y_pred, average='micro', beta=2) )
    print("Sample of y_true : {}, \n Sample of y_pred : {}".format(y_true[:5], y_pred[:5]))
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    accuracy  = accuracy_score(y_true,y_pred)
    print("accuracy : {}".format(accuracy))
    print("accuracy : {}".format(accuracy_score(y_true,y_pred)))
    print("F1 Score: {}".format(f1_score(y_true,y_pred, average='micro')))
    print("Recall score: {}".format(recall_score(y_true, y_pred, average='micro')))
    print("Precision score: {}".format(precision_score(y_true,y_pred, average='micro')))
    print(classification_report(y_true, y_pred))
    print("evaluation is done.")
    model.train()

    return accuracy



def eval_4_multi(y_pred, y_true, label_dict):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import confusion_matrix
    
    if torch.cuda.is_available():
        y_true = torch.cat(y_true).cpu().numpy().astype(int)
        y_pred = torch.cat(y_pred).cpu().numpy().astype(int)
    else:
        y_true = torch.cat(y_true).numpy().astype(int)
        y_pred = torch.cat(y_pred).numpy().astype(int)


    y_true = [label_dict["index2label"][i] for i in y_true]
    y_pred = [label_dict["index2label"][i] for i in y_pred]


    print("F-2 score : ", fbeta_score(y_true, y_pred, average='micro', beta=2) )
    print("Sample of y_true : {}, \n Sample of y_pred : {}".format(y_true[:5], y_pred[:5]))
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    accuracy  = accuracy_score(y_true,y_pred)
    print("accuracy : {}".format(accuracy))
    print("accuracy : {}".format(accuracy_score(y_true,y_pred)))
    print("F1 Score: {}".format(f1_score(y_true,y_pred, average='micro')))
    print("Recall score: {}".format(recall_score(y_true, y_pred, average='micro')))
    print("Precision score: {}".format(precision_score(y_true,y_pred, average='micro')))
    print(classification_report(y_true, y_pred))
    print("evaluation is done.")


    return accuracy




def save_restricted_model(model, filepath):
    import torch
    torch.save(model, filepath)
    # Later, load with below line
    # model = torch.load(filepath)
    return None

def save_model_4_retraining(model, file_path, checkpoint):
    """
    Case 2: Save model to resume training later
    If you need to keep training the model that you are about to save
    you need to save more than just the model.
    You also need to save the state of the optimizer, epochs, score, etc.
    You would do it like this:

    ImageNet example,
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def load_pretrained(model, pretrained_dict):
        model_dict = model.state_dict()
        # filter out unmatch dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

    if checkpoint_file is not None:
        print('loading checkpoint_file {}'.format(checkpoint_file))
        cp = torch.load(checkpoint_file)
        load_pretrained(optimizer, cp['optimizer'])
        load_pretrained(state_dict, cp['state_dict'])
        trained_episodes = cp['epoch']
    """
    import torch
    torch.save(state, filepath)

    return None



def fixed_size_Pool1d(X, pool_type="average", batch_first=False, filter_size_split=2):
    """
    X with batch_fist False is, output of lstm. from ( output, (ht, ct) = lstm(x_embed) )
    batch_fist is False : (seq_len, batch_size, lstm_out_dim)
    batch_fist is True  : (batch_size, seq_len, lstm_out_dim)
    """
    import torch.nn as nn

    # Step1 - just average Pooling
    if not batch_first: X = X.t() # (L, B, D) -> (B, L, D)
    elif batch_first : pass

    output_dim = X.size(2)

    if pool_type is "average":
        ap = nn.AvgPool1d(output_dim, stride=output_dim)
        out = ap(out)
    elif pool_type is "max":
        mp = nn.MaxPool1d(output_dim, stride=output_dim)
        out = mp(out)

    # Step2 - Make output shape same.
    length = out_ap.size(2)
    fs = filter_size_split
    filter_size = int(length/fs) if (length/fs)%2 ==0 else int(length/fs)+1
    stride_size = int(filter_size/fs)

    if pool_type is "average":
        ap = nn.AvgPool1d(filter_size, stride=stride_size)
        out = ap(out)
    elif pool_type is "max":
        mp = nn.MaxPool1d(filter_size, stride=stride_size)
        out = mp(out)

    return out


def convert_word_dict(word_dict_obj):
    """
    covnert object word_dict to dictionary format.

    How to use:
    word_dict = convert_word_dict(word_dict) # obj->dict
    """
    temp_dict = {}
    temp_dict["word2index"] = word_dict_obj.dictionary.word2idx
    temp_dict["index2word"] = word_dict_obj.dictionary.idx2word
    word_dict = temp_dict
    del temp_dict
    # print(word_dict["word2index"]["cat"])
    return word_dict






def DEPRECATE_custom_dataloader(X, y, label_dict, batch_size=10):
    """
    use customized DataLoader

    How to use:

    train_file_path = ".../cleaned_train_shuffled.txt"
    test_file_path  = "".../cleaned_test_shuffled.txt"

    TARGET_PROPORTION_LIST = [0.5] # only want to use 5% of labeled dataset.
    for proportion in TARGET_PROPORTION_LIST:
        X_train, X_test, y_train, y_test, label_dict = get_proportion_labeled_data(train_file_path, test_file_path, target_proportion=proportion)

    batch_size = 10
    trainloader = custom_dataloader(X_train, y_train, label_dict, batch_size=batch_size)
    testloader  = custom_dataloader(X_test, y_test, label_dict, batch_size=batch_size)
    """
    yield_list = list()
    for text, label in zip(X, y):
        label = label_dict["index2label"][label]
        yield_list.append(label+' '+text)
        if len(yield_list) >= batch_size:
            line = yield_list
            yield_list = list() # flush yield_list
            yield line
            
            
def custom_dataloader(X, y, label_dict, batch_size=10, complexity=5):
    """
    use customized DataLoader

    How to use:

    train_file_path = ".../cleaned_train_shuffled.txt"
    test_file_path  = "".../cleaned_test_shuffled.txt"

    TARGET_PROPORTION_LIST = [0.5] # only want to use 5% of labeled dataset.
    for proportion in TARGET_PROPORTION_LIST:
        X_train, X_test, y_train, y_test, label_dict = get_proportion_labeled_data(train_file_path, test_file_path, target_proportion=proportion)

    batch_size = 10
    trainloader = custom_dataloader(X_train, y_train, label_dict, batch_size=batch_size)
    testloader  = custom_dataloader(X_test, y_test, label_dict, batch_size=batch_size)
    """
    import numpy as np
    X_y_combined = list(zip(X, y))
    for i in range(complexity):
        np.random.shuffle(X_y_combined)
    
    yield_list = list()
    for text, label in X_y_combined:
        label = label_dict["index2label"][label]
        yield_list.append(label+' '+text)
        if len(yield_list) >= batch_size:
            line = yield_list
            yield_list = list() # flush yield_list
            yield line
            
            
            
def report(loss_list, nhid_linear, learning_rate, save_dir_path, description=''):
    """
    Simple report to save loss curve. 15th March.

    How to use:
    report(loss_list, nhid_linear, learning_rate, './graph/')
    """
    import matplotlib.pyplot as plt
    import pickle, datetime
    current_time =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    title = "{}units_{}lr_{}".format(nhid_linear, learning_rate, current_time)
    title += '_'+ description

    print("Saving {}".format(title))

    file_path = save_dir_path+"/"+title
    with open(file_path+'.p', 'wb') as f:
        pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Load with - pickle.load(f)

    x = range(len(loss_list))
    y = loss_list
    plt.plot(x, y)
    plt.title("{}units_{}lr_{}".format(nhid_linear, learning_rate, description))
    # jpg occur error.
    plt.savefig(file_path+'.png')
    # plt.show()

    return None




def pool_4_nlp(inputs, pool_type="average", batch_first=False, window_size=5, pad=0, nunit=250, stride=1):
    """
    X with batch_fist False is, output of lstm. from ( output, (ht, ct) = lstm(x_embed) )
    batch_fist is False : (seq_len, batch_size, lstm_out_dim)
    batch_fist is True  : (batch_size, seq_len, lstm_out_dim)
    """
    from torch.nn import functional as F

    #=================== Step1 - just average Pooling ===================
    if not batch_first: inputs = inputs.t() # (L, B, D) -> (B, L, D)
    elif batch_first : pass

    if pool_type is "average":
        out = F.adaptive_avg_pool1d(inputs, 1)
    elif pool_type is "max":
        out = F.adaptive_max_pool1d(inputs, 1)
    # ===================================================================


    #=================== Step2 - Apply Real Pooling =====================
    batch_size = out.size(0)
    seq_len    = out.size(1)

    out_reshape = out.view(batch_size, 1, seq_len)


    # We made window size as same as stride(gap) value.
    filter_size     = window_size # window size is fixed. or, should we apply variable filter_size(window size)?
    stride_interval = stride

#     print("Filter size : {}, stride interval : {}".format(filter_size, stride_interval))
#     print("Output reshape : {}".format(out_reshape.size()))


    #=================== Step3 - Make all size of pooled matrix same. =====================
    current_units = out_reshape.size(2)

    while(current_units > nunit):
        # More pooling
        if pool_type is "average":
            out_pooled = F.avg_pool1d(out_reshape, filter_size, stride=stride_interval)
        elif pool_type is "max":
            out_pooled = F.max_pool1d(out_reshape, filter_size, stride=stride_interval)

        out_reshape   = out_pooled
        current_units = out_pooled.size(2)
#         print("running, current_units : ", current_units)

    if current_units != nunit:
        # Padding
        # Below line do padding with 0. left for nothing(0), right for (nunit-current_units) padding
        result = F.pad(out_reshape, [0,nunit-current_units], 'constant', pad) # It works on GPU directly.
    else:
        result = out_reshape

    result = result.squeeze()
    return result


class save_model_obj():
    def __init__(self):
        self.previous_current_loss = 10**10
        self.previous_val_loss    = 10**10
        self.previous_val_acc     = -999

    def only_best(self, model, current_loss, val_loss, val_acc, file_name='', only_acc=False):
        import pickle, datetime
        import torch
        # How to load
        # model.load_state_dict()

        if current_loss < self.previous_current_loss and val_loss < self.previous_val_loss and val_acc > self.previous_val_acc:
            # current_time =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            file_name = 'best_model_' + file_name +'.pt'
            torch.save(model.state_dict(), file_name)

            print("*"*59)
            print("Saved BEST MODEL ACC+VAL {}".format(file_name))
            print("*"*59)

            self.previous_current_loss = current_loss
            self.previous_val_loss     = val_loss
            self.previous_val_acc      = val_acc
            return True
        else:
            print("Could not find best model with (loss, acc)")


        if only_acc:
            if val_acc > self.previous_val_acc:
                file_name = 'best_model_' + file_name +'.pt'
                torch.save(model.state_dict(), file_name)
                print("*"*59)
                print("Saved ACCURACY only : {}, acc: {}".format(file_name, round(val_acc,2)))
                print("*"*59)
                self.previous_val_acc      = val_acc
                return True
            else:
                print("Could not find best ACCURACY model ")
                return False
                
        print('Could not find any best point')
        return False
    
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
    
    
    
    
def evaluate(test_loader, model_list, word_dict_list, label_dict, args):
    from smnlp_tl import eval_4_multi
    from smnlp_general import report_matrix
    import gc, time, sys
    
    # ========== Only train Main model and classifier =========
    # All models are not trainable.
    for i, model in enumerate(model_list):
        model_list[i] = model.eval()
        del model; gc.collect()
    # ===========================================================

    cuda = args.cuda

    y_true, y_pred = list(), list()

    test_instance = 0
    total_instance = int(len(test_loader)*args.batch_size_eval)
    test_limit	= 10**15 if args.test_instance ==0  else args.test_instance
    training_hidden_state = args.training_hidden_state
    print("===================== Start evaluation =====================\n")


    for lines in test_loader:
        
        start_time_batch = time.time()
        # =========== Validation/Count trained instances ============
        # elif args.cross_val:
        # 	next(cross_val_point)
        if test_instance >= test_limit:
            break
        test_instance += args.batch_size_eval   # Save this to check how many instances are trained.
        # =========== Validation ============


        seq_tensor_list   = []

        for word_dict in word_dict_list:
            seq_tensor, labels_tensor, seq_length = oneshot_padding(lines, label_dict, word_dict, volatile=True, unk_symbol="<unk>", eos_symbol="<eos>")
            if cuda:
                seq_tensor = seq_tensor.cuda(async=True)


            seq_tensor = seq_tensor.t().contiguous()
            seq_tensor_list.append(seq_tensor)

        seq_tensor_list = torch.stack(seq_tensor_list, 0)

        if cuda: labels_tensor = labels_tensor.cuda(async=True)
        if cuda: seq_length    = seq_length.cuda(async=True)


        # Only calculate [main, support]-LM blocks not ModelClassifier.
        outputs = []
        for i, (model, seq_tensor) in enumerate(zip(model_list, seq_tensor_list)):
            output, _ = model(seq_tensor, seq_length)
            outputs.append(output)
            del model; torch.cuda.empty_cache() #gc.collect()?



        # ============== Forward concatenated a layer into model classifier ===========
        cat_input = torch.cat(outputs, 1)

        # use model_cat for classifier.
        output = model_list[-1](cat_input)
        _, predicted = torch.max(output, 1)

        target = labels_tensor.squeeze()

        y_true.append(target.data)
        y_pred.append(predicted.data)
        del [_, cat_input, output, predicted]
        gc.collect(); torch.cuda.empty_cache()
        
        onebatch_time = "one batch taskes {} second".format(round(time.time() - start_time_batch, 2))
        sys.stdout.write("\r current batch -> {}/{} | {} percent | {}".format(test_instance, total_instance, round(100*(test_instance/total_instance), 3), onebatch_time))
        sys.stdout.flush()
        
    return y_pred, y_true