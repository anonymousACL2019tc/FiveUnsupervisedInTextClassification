## How to use:

$ python LM_block_main.py --static_val --wdrop 0.5 --nhid 400 --lr 0.0001 --adam  --epoch 15 --test_instance 0 --log_file_name agnews_log --best_model agnews_best --batch_size 2 --batch_size_eval 360  --train_data "/TODO_DATA_DIRECTORY/Target_agnews/agnews_train"$instance".txt" --cuda_device 0 --test_data "/TODO_DATA_DIRECTORY/Target_agnews/agnews_test.txt" --target_dict "/TODO_PRETRAIN_DIRECTORY/agnews_word_dict.obj" --target_checkpoint "/TODO_PRETRAIN_DIRECTORY/agnews.checkpoint" --test_instance 0
