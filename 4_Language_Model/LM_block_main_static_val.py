# Written in Python3
# Sung-Min, Yang. studying Language Technology in University of Gothenburg.
# 2018-04-06 v1


# Load libaries related to [Deep learning]
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

# Import standard python libraries 
import re, gc, pickle, time, argparse, random, sys


# Import custom libraries
from smnlp_tl import *
from smnlp_general import *
from sm_model import LMmodel
from sm_model import ClassifierModel


parser = argparse.ArgumentParser(description='input arguments')

parser.add_argument('--train_data', type=str, default="/home/sung_min/experiment_thesis/public_data/Target_agnews/agnews_train10.txt", help='train data path')
parser.add_argument('--test_data', type=str, default="/home/sung_min/experiment_thesis/public_data/Target_agnews/agnews_test.txt", help='test data path')
parser.add_argument('--target_dict', type=str, default="/home/sung_min/experiment_thesis/LM_training/best_weights/agnews_word_dict.obj", help='target dictionary path')
parser.add_argument('--target_checkpoint', type=str, default="/home/sung_min/experiment_thesis/LM_training/best_weights/agnews2April.checkpoint", help='target checkpoint')


parser.add_argument('--support_LMs_dict', default=[], nargs='+', help='''type list of support language model path(checkpoint/state_dict)
    How to use:
    --support_LMs_dict "...path/to/yelp_word_dict.obj" "...path/to/yahoo_word_dict.obj"  ".../*.obj" => it becomes array
    --support_LMs_dict "/home/sung_min/experiment_thesis/LM_training/best_weights/wiki_word_dict.obj" => use only one support LM
    ''')
parser.add_argument('--support_LMs', default=[], nargs='+', help='''type list of support language model path(checkpoint/state_dict)
    How to use:
    --support_LMs ...path/to/yelp.checkpoint
    --support_LMs '/home/sung_min/experiment_thesis/LM_training/best_weights/checkPoint_wiki103March28'
    ''')


parser.add_argument('--log_file_name', type=str, default="", help='name of file you want to save')
parser.add_argument('--output_path', type=str, default="./log_multi_models/", help='where is the output path?')
parser.add_argument('--best_model', type=str, default="", help='name of best model')

parser.add_argument('--lr', type=float, default='120', help='learning rate')
parser.add_argument('--batch_size', type=int, default='30', help='batch_size')
parser.add_argument('--batch_size_eval', type=int, default='80', help='batch_size')
parser.add_argument('--nhid', type=int, default=100, help='number of classifier hidden units')
parser.add_argument('--wdrop', type=float, default=0.5, help='number of classifier hidden units')
parser.add_argument('--adam', action="store_true", help='use Adam optimizer')
parser.add_argument('--cosine_annealing', action="store_true", help='use cosine annealing')
parser.add_argument('--training_hidden_state', action="store_true", help='use training_hidden_state')
parser.add_argument('--pool', default="mixed_pool", help='which pool method you want use')
parser.add_argument('--test_instance', type=int, default=0, help='number of test instance')
parser.add_argument('--val_percent', type=float, default=0.1, help='validation percent in train set')
# parser.add_argument('--cross_val', type=int, default=0, help='Cross validation, how many folds?')
parser.add_argument('--static_val', action="store_false", help='shuffle validation set? true or not')


parser.add_argument('--cuda', action="store_false", help='use cuda by default, if you type this, it means dont want to use cuda')
parser.add_argument('--seed', type=int,  default=2212, help='set seed')
parser.add_argument('--epoch', type=int,  default=10, help='specify epoch')
parser.add_argument('--early_stop', type=int,  default=4, help='early stop patience')
parser.add_argument('--cuda_device', type=int, default=0, help='Specify cuda device')


args = parser.parse_args()



# ========== check parser error ==========
if args.training_hidden_state:
    assert args.batch_size == args.batch_size_eval, "if you are using training hidden state, batch size for train/eval must be same"

assert args.log_file_name != "", "log file name must be provided"
assert len(args.support_LMs)==len(args.support_LMs_dict), "must be same"


  
# =========== In order to reproduce same result. below 2 lines ===========
# Set the random seed manually for reproducibility.
torch.backends.cudnn.benchmark = False # In case, facing problem CUDNN_STATUS_INTERNAL_ERROR

np.random.seed(args.seed)
torch.manual_seed(args.seed)
try : torch.cuda.manual_seed(args.seed)
except : pass
# ========================================================================



# ==================== Ask user to use cuda ====================
if torch.cuda.is_available():
    print("There are {} GPU exist".format(torch.cuda.device_count()))

    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        user_input = input("Are you sure you don\'t want  to use cuda? y/n")
        
        if user_input.lower()=="n":
            print("you choose to use CPU")
        elif user_input.lower()=="y":
            args.cuda = True
            user_input=input("which cuda device you want to use? {} device(s) exist".format(torch.cuda.device_count()))
            torch.cuda.set_device(int(user_input))
            print("You want to use cuda device {},  current cuda set device  {}".format(user_input, torch.cuda.current_device()))
    else:
        torch.cuda.set_device(args.cuda_device)
        print("You want to use cuda device {},  current cuda set device  {}".format(args.cuda_device, torch.cuda.current_device()))





# ========================== Load data with DataLoader ========================

support_LMs		= args.support_LMs
support_LMs_dict   = args.support_LMs_dict

# from_scratch  = False  # training from scratch.


train_file_path = args.train_data
test_file_path  = args.test_data

target_dict_path  = args.target_dict
target_checkpoint_path = args.target_checkpoint


support_LMs_pt_dict = []

checkpoint_list = [target_checkpoint_path] + support_LMs
# checkpoint_list became [target.pt, supportLM1.pt, supportLM2.pt, ...]
word_dict_list  = [target_dict_path] + support_LMs_dict
# word_dict_list became [target_word_dict.obj, supportLM1_word_dict.obj, supportLM2_word_dict.obj, ...]


# ========================== Generate trainloader to build label_dict ===========================
batch_size = args.batch_size

# import multiprocessing
# max_cpu = 4
# num_workers = int(max_cpu/2) if max_cpu/2 >= 2 else 1
num_workers = 4


trainloader = get_train_loader(train_file_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader  = get_train_loader(test_file_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)

label_dict = get_label_dict(dataloader=trainloader)


# ========================== Load pickled word dictionary ==========================
for i, word_dict in enumerate(word_dict_list):
    word_dictionary = get_word_dict(word_dict)		   # Load pickle file
# ======== util function for obtaining word dictionary(dict structure) from class structure =============
    word_dictionary = convert_word_dict(word_dictionary) # convert class type to dict type 
    word_dict_list[i] = word_dictionary
    del word_dict, word_dictionary; gc.collect()

# DEPRECATED
#word_dict_target = get_word_dict(word_dict_path_target, check_word2index=None, check_index2word=20000) # <---- it does not include <EOS>, but eos, eoc: end of sentence, contract




# ======== Configuration from AWD-LSTM. ========
dropout  =  0.4 
dropouth =  0.3 
dropouti = 0.65 
dropoute =  0.1 
wdrop	=  args.wdrop
ninp	 =  emsize   =  400 
nhid	 = 1150 
nlayers  =	3
lr	   =   30 
clip	 = 0.25 
bptt	 =   70 
tied	 = True
rnn_type = "LSTM"
weight_decay = 1.2e-6

# ======== New configuration for text classification ========
ntoken_list  = [len(word_dict['word2index']) for word_dict in word_dict_list]
nclass   = len(label_dict['label2index'])
batch_size = batch_size
nhid_linear = args.nhid
pooling = args.pool #"normal_pool"# "last_hidden" #"smart_pool", multi_hiddn_pool
batch_norm = True
dropout_additional = 0.2
relu = False
final_out_dim = int(nhid*(nlayers-1)+ninp)*len(checkpoint_list) if pooling=="multi_hidden_pool" else emsize if pooling=="last_hidden" else 2000*len(checkpoint_list) # or len(word_dict_list)


trainloader = get_train_loader(train_file_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader  = get_train_loader(test_file_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# for data in trainloader:
#     print("Sample : ", data[:1])
#     seq_tensor, labels_tensor, seq_length = oneshot_padding(data[:3], label_dict, word_dict_list[0], unk_symbol="<unk>", eos_symbol='<eos>')
#     break



# ================= Create main LM model and support LMs. =================
model_list = []
for ntoken in ntoken_list:
    model_list.append(
            LMmodel(rnn_type, ntoken, ninp, nhid, nlayers, dropout, dropouth, dropouti, dropoute, wdrop, tied, pooling)
                     )
model_classifier  = ClassifierModel(final_out_dim, nclass, nhid_linear, batch_norm, dropout_additional, relu)
# model_list = ['model_target', 'model_wiki', ....]



# ================= Load pretrained LMs =================
# for model, checkpoint_path in zip(model_list, checkpoint_list):
for i, (model, checkpoint_path) in enumerate(zip(model_list, checkpoint_list)):
    # pretrained_dict = torch.load(checkpoint_path)['state_dict']  # <---- not working.
    pretrained_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['state_dict']
    
    load_pretrained_model(model, pretrained_dict)
    del model, pretrained_dict; gc.collect()
    #del model, pretrained_dict; gc.collect()


# ================= After finish load pretrained LMs, add model classifier into model_list =================
model_list += [model_classifier]


# =========== Make it sure to target model weights are all trainable
model_list[0] = freeze_layers(model_list[0]) # all trainable




# ====================== Check how many instances/batches exist and validation instance information ======================
trainloader = get_train_loader(train_file_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Get total number of batches
total_instance = sum([batch_size for _ in trainloader])
num_batches = int(total_instance/batch_size)
print("Total batches : ", num_batches)
print("Total instances : ", total_instance)
trained_instance = total_instance
instance_per_class = int(total_instance/nclass)




# ---------- Make validation split-point for Cross validation. ----------
# if args.cross_val: 
#     print("You chose Cross validation {} folds".format(args.cross_val))
#     validation_point = (i for i in range(int(total_instance/args.cross_val))) # 100 -> 10
#     # validation_point_list = (0, 10, 20, ... 90) <--- generator



validation_percent = args.val_percent
validation_point   = int(num_batches*(1-validation_percent))
print("Validation percent : {} \nValidation point batch index : {}".format(validation_percent, validation_point))
# =========================================================================================================================







# ================================= Configuration for training  =================================
batch_size = batch_size
cuda   = args.cuda
# custom_data_loader = custom_data_loader
evaluation_mode  = True

learning_rate = args.lr #0.003
revive_learning_rate_point=0
clip = 0.01 # gradient clip # try clip the gradient #@TODO

cosine_annealing = args.cosine_annealing
adam = args.adam

training_hidden_state = args.training_hidden_state
best_hidden = False


save_model = save_model_obj()

# DEPRECATED
# randomly_freeze = False
# unfreeze_gradually = False
# different_lr_layer = False
# ===============================================================================================


criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=3e-1)
# If anyone wants to see if weights change faster, set learning rate as 30.




# =========== Change model mode to GPU/CPU ===========
if cuda:
    for i, model in enumerate(model_list):
        model_list[i] = model.cuda()
        del model; gc.collect()
else:
    for i, model in enumerate(model_list):
        model_list[i] = model.cpu()
        del model; gc.collect()
        
        
        
        
if adam:
    print('------ Using Adam optimizer ------')
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.7, 0.9), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(list(model_list[0].parameters()) +
                           list(model_list[-1].parameters()),
                           lr=learning_rate, weight_decay=weight_decay) # betas=(0.7, 0.9),
else:
    print('------ Using SGD optimizer ------')
#          from functools import reduce
#          import operator
#          model_param = [list(filter(lambda p: p.requires_grad, m.parameters())) for m in model_list]
#          model_param = reduce(operator.concat, model_param)
#          import pdb;pdb.set_trace()
#          optimizer = optim.SGD(model_param, lr=learning_rate, weight_decay=weight_decay)

    # Only update model_target
    optimizer = optim.SGD(list(model_list[0].parameters()) +
                          list(model_list[-1].parameters()),
                          lr=learning_rate, weight_decay=weight_decay)






# ============== Tools for loss, acc measuring ============== 
batch_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
loss_avg_list = []
loss_val_list = []
val_acc_list  = []
# =========================================================== 

early_stop_patience = 0
epoch = 1
while(epoch<=args.epoch):
    print("-------------New EPOCH -------------")
    print("currently, {} epoch".format(epoch))
    start_time = time.time()

#     if unfreeze_gradually:
#         freeze_range = [(0,12), (0,8), (0,4), None]
#         model = freeze_layers(model, indice=freeze_range[epoch-1 if epoch<=len(freeze_range) else -1])



    # ========== Only train Main model and classifier =========
    # Only main LM-block and model-classifier are trainable.
    for i, model in enumerate(model_list):
        if i==0 or i==(len(model_list)-1):
            model_list[i] = model.train()
        else: model_list[i] = model.eval()
        
        del model; gc.collect()
    # ===========================================================



    if training_hidden_state:
        print("----- Generating hidden state for training -----")
        for model in model_list:
            hidden = model.init_hidden(batch_size)
            del model; gc.collect()
            break #model_list[0] is only matter, just break out after first loop.
            

    

    validation_loader = []
    trained_instance_num = 0
    trainloader = get_train_loader(train_file_path, batch_size=batch_size, shuffle=args.static_val, num_workers=num_workers)
    # for i in range(args.cross_val):
    for batch_cnt, lines in enumerate(trainloader):
        start_time_batch = time.time()
        
        # =========== Validation/Count trained instances ============
        # elif args.cross_val:
        # 	next(cross_val_point)
        if batch_cnt+1 >= validation_point and evaluation_mode:
            validation_loader.append(lines)
            continue
        trained_instance_num += batch_size   # Save this to check how many instances are trained.
        # =========== Validation ============
        
        seq_tensor_list   = []
        
        for i, word_dict in enumerate(word_dict_list):
            if i==0:
                seq_tensor, labels_tensor, seq_length = oneshot_padding(lines, label_dict, word_dict, unk_symbol="<unk>", eos_symbol="<eos>")
            else:
                seq_tensor, labels_tensor, seq_length = oneshot_padding(lines, label_dict, word_dict, volatile=True, unk_symbol="<unk>", eos_symbol="<eos>") #<--------------------------- @TODO
                
            if cuda:
                seq_tensor = seq_tensor.cuda(async=True) # async=True
                
                
            seq_tensor = seq_tensor.t().contiguous()
            seq_tensor_list.append(seq_tensor)
        
        
            
        seq_tensor_list = torch.stack(seq_tensor_list, 0)
        
        if cuda: labels_tensor = labels_tensor.cuda(async=True) # async=True
        if cuda: seq_length    = seq_length.cuda(async=True)    # async=True
        
        
        # Only calculate [main, support]-LM blocks not ModelClassifier.
        outputs = []
        for i, (model, seq_tensor) in enumerate(zip(model_list, seq_tensor_list)):
            output, _ = model(seq_tensor, seq_length)
            outputs.append(output)
            del model; torch.cuda.empty_cache() #gc.collect()?

        
        
        # ============== Forward concatenated a layer into model classifier ===========		
        cat_input = torch.cat(outputs, 1)

        # use model_cat for classifier.
        # Ensure vaiable can be traiable for classifier.
        if args.support_LMs : cat_input.volatile = False #<--------------------------- @TODO
        
        output = model_list[-1](cat_input)

        target = labels_tensor.squeeze()
        loss = criterion(output, target)

        inputs  = seq_tensor.data
        losses.update(loss.data[0], inputs.size(0))
        # ============ loss calcaulation ===============


        # ============ grad udpate ================
        optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm(model_list[0].parameters(), clip)
            torch.nn.utils.clip_grad_norm(model_list[-1].parameters(), clip)
        optimizer.step()
        # ============ grad udpate ================


        # =========== Repackage ============
        if training_hidden_state:
            hidden = repackage_hidden(hidden)
        # =========== =========== ============


        if cosine_annealing:
            origin_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = 0.5*(1+np.cos(np.pi*batch_cnt/num_batches))*learning_rate

        onebatch_time = "one batch taskes {} second".format(round(time.time() - start_time_batch, 2))
        sys.stdout.write("\rTrained batch -> {}/{} | {} percent | {}".format(trained_instance_num, trained_instance, round(100*trained_instance_num/((1-validation_percent)*trained_instance), 3), onebatch_time))
        sys.stdout.flush()
        
    print("one epoch training is over, 1 epoch taskes {} second".format(round(time.time() - start_time, 2)));
    start_time = time.time()


    if evaluation_mode:
        print("Start validation test, percent : {}".format(validation_percent))
        print("Trained instances for 1 epoch : {}".format(trained_instance_num))
        print("Number of validation instances for test : {}".format(len(validation_loader)*batch_size))
        print("====================================\n\n")
        print("loss val is {}, loss avg : {}".format(losses.val, losses.avg))

        
        # Using validation laoder
        y_pred, y_true = evaluate(validation_loader, model_list, word_dict_list,label_dict, args)
        accuracy = eval_4_multi(y_pred, y_true, label_dict)
        best = save_model.only_best(model_list[0], losses.avg, losses.val, accuracy, file_name=args.best_model, only_acc=True)
        
        
        if best:
            early_stop_patience = 0
            torch.save(model_list[-1].state_dict(), args.best_model+'.cat')

        elif not best:
            early_stop_patience += 1
        

    

        # =============== Save graph ===============
        loss_avg_list.append(losses.avg)
        loss_val_list.append(losses.val)
        val_acc_list.append(accuracy)
        # =============== Save graph ===============


        # Verify train mode(even though, eval function has this line.)
        model_list[0].train() 
        model_list[-1].train()
        
        
        if early_stop_patience >= args.early_stop:
            print("\n ---------Early stop by patience {}-------- \n".format(early_stop_patience))
            break    
            

    epoch+=1
    print("one epoch evaluation is over, evaluation taskes {} second \n\n".format(round(time.time() - start_time, 2)))
    
        
    # new learning rate for every epoch.
    if not cosine_annealing and not adam:
        origin_lr = optimizer.param_groups[0]['lr']
        if origin_lr < revive_learning_rate_point:
            optimizer.param_groups[0]['lr'] = learning_rate/2 # Revive learning rate
            print('****************** REVIVE Learning rate ****************** lr : {}\n'.format(learning_rate/2))
        else : optimizer.param_groups[0]['lr'] = origin_lr/2




# =================== Save graph into JSON ===================
import json
graph_dict = {"learning_rate" : args.lr,
             "batch_size"	: args.batch_size,
             "epoch"		 : args.epoch,
             "nhid"		  : args.nhid,
             "adam"		  : args.adam,
             "cosine_annealing" : args.cosine_annealing,
             "training_hidden_state" : args.training_hidden_state,
             "seed"		  : args.seed,
             "wdrop"		 : args.wdrop,
             "loss_avg" : loss_avg_list,
             "loss_val" : loss_val_list,
             "val_acc" : val_acc_list,
             "trained_instance": trained_instance,
             "instance_per_class": instance_per_class,
             "pooling":args.pool}


graph_file_path = args.output_path + args.log_file_name+'.graph'



import os
if os.path.isfile(graph_file_path):
    mode = 'a'
else: mode = 'w'

with open(graph_file_path, mode, encoding='utf8') as f:
    json.dump(graph_dict, f)
    f.write('\n')
# =================== Save graph into JSON ===================	




# ==================== Start evaluation ====================
load_additional = False

if load_additional:
    with open('label_dict.target', 'rb') as f:
        label_dict = pickle.load(f)

best_model      = torch.load('best_model_'+args.best_model+'.pt')
best_model_cat  = torch.load(args.best_model+'.cat')
load_pretrained_model(model_list[0], best_model)
load_pretrained_model(model_list[-1], best_model_cat)


# ========= DEPRECATED =======
# eval(model, testloader, label_dict, word_dict, batch_size, trained_hidden=best_hidden, cuda=True, trial=1000000, unk_symbol="<unk>", eos_symbol="<eos>")



testloader  = get_train_loader(test_file_path, batch_size=args.batch_size_eval, shuffle=True, num_workers=num_workers)
y_pred, y_true = evaluate(testloader, model_list, word_dict_list, label_dict, args)
accuracy = eval_4_multi(y_pred, y_true, label_dict)


title = args.log_file_name
description = str(instance_per_class)+"instance_per_class"+"_lr_"+str(args.lr)+"_adam_"+str(args.adam)+"_cosine_"+str(args.cosine_annealing)+"_nhid_"+str(args.nhid)+"_wdrop_"+str(args.wdrop) + "_trainHidden_"+str(args.training_hidden_state)+"_multiLM_"+str(args.support_LMs)+"_pooling_"+str(pooling)+"_clip_"+str(clip)

report_matrix(args.output_path, description, title, y_true, y_pred, label_dict=label_dict, append=True, torch=True)



