def get_reviews_from_json(json_data, strip=True):
    num_reviews = len(json_data["Reviews"])
    review_list = list()

    for i in range(num_reviews):
        try :
            rating = json_data["Reviews"][i]["Ratings"]["Overall"]
            content  = json_data["Reviews"][i]["Content"]
            if strip: title  = json_data["Reviews"][i]["Title"].strip("\“\”")
            else : title  = json_data["Reviews"][i]["Title"]

            review_list.append(rating + " " + content + " " + title)
        except : continue
    return review_list




def get_clean_data_from_csv(filename, train_or_test='train'):
    from smnlp_general import convert_proper_noun
    from smnlp_general import sm_tokenize
    from smnlp_general import check_sample_lines
    from smnlp_general import shuffle_text_file

    from pandas import read_csv
    import csv
    train_or_test = 'test' # train or test

    path = './Target_yelp_f'
    text_path = path + "/"+str(train_or_test)+".csv"
    target_path = path+"/cleaned_"+str(train_or_test)+".txt"

    f_write = open(target_path, 'w', encoding='utf8')

    print(text_path, target_path)

    with open(text_path, 'r', encoding='utf8') as f:
        lines = csv.reader(f)
        for line in lines:
            score, text = line
            line = convert_proper_noun(text)                    # Convert {2,3,4} grams of proper nouns : New York-> New_York
            line = sm_tokenize(line, remove_stopwords=True, remove_more=[r'\\n'])     # tokenize, make line cleaner.

            line = ' '.join([score, line])
            if line.endswith('\n'): f_write.write(line)
            else : f_write.write(line+'\n')
    f_write.close()


    # make shuffled file
    shuffle_text_file(target_path)

    return None




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


def get_proportion_labeled_data(train_file_path, test_file_path, target_proportion=0.05):
    """
    Treat labeled data as labeled + unlabeled.
    By setting target_propotion, we can treat train data as only labeled one,
    and treat rest of them as unlabeled one

    How to use:
    train_file_path = "/home/sung_min//experiment_thesis/public_data/Target_ag_news_csv/cleaned_train_shuffled.txt"
    test_file_path  = "/home/sung_min//experiment_thesis/public_data/Target_ag_news_csv/cleaned_test_shuffled.txt"
    portion = 0.05
    X_train, X_test, y_train, y_test, label_dict = get_proportion_labeled_data(train_file_path, test_file_path,
    target_proportion=proportion)
    """

    from sklearn.model_selection import train_test_split
    from smnlp_general import get_tag_and_training_data
    import gc
    if target_proportion == 1.0 :
        y_train, X_train, label_dict = get_tag_and_training_data(train_file_path, label_dict=False)
        y_test,  X_test,  label_dict = get_tag_and_training_data(test_file_path, label_dict=label_dict)
        return X_train, X_test, y_train, y_test, label_dict
    else:
        labels_train, train_data, label_dict = get_tag_and_training_data(train_file_path, label_dict=False)
        y_test, X_test, label_dict = get_tag_and_training_data(test_file_path, label_dict=label_dict) # use same label_dict from train set

        _, X_train, _, y_train = train_test_split(train_data, labels_train, test_size=target_proportion, shuffle=True)
        del _
        gc.collect()
        return X_train, X_test, y_train, y_test, label_dict










def get_tfidf_weights(list_of_texts, encoding='utf-8', decode_error='strict', strip_accents=None,
 lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None,
 token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None,
 vocabulary=None, binary=False, dtype='<class <numpy.int64>', norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
	"""
	This function returns tfidf weights.
	In : ["doc1", "doc2", "doc3", ...]
	Out : {'a': 0.12, 'cat':1.32, 'lion': 3.22, ...}

	e.g., Feed input of ['sentence1', 'sentence2', ...]
	"""
	from sklearn.feature_extraction.text import TfidfVectorizer

	tfidf_vec = TfidfVectorizer(token_pattern=r'\b\w+\b', min_df=1)
	X = tfidf_vec.fit_transform(list_of_texts)
	idf = tfidf_vec.idf_

	tfidf_wiehgt = dict(zip(tfidf_vec.get_feature_names(), idf))

	return tfidf_wiehgt

def convert_proper_noun(sentence):
    """
    Find and convert 4gram-3gram-2gram of proper noun
    e.g.,
    4gram : Best Western Pioneer Square -> Best_Western_Pioneer_Square
    3gram : Pike Place Market           -> Pike_Place_Market
    2gram : New York                    -> New_York

    RETURN : whole sentence with converted proper nouns.
    e.g.,
    INPUT  sentence : this is New York
    OUTPUT sentence : this is New_York
    """
    import re
    proper_noun_tuples = re.findall(r'([A-Z]\w+ [A-Z]\w+ [A-Z]\w+ [A-Z]\w+)|([A-Z]\w+ [A-Z]\w+ [A-Z]\w+)|([A-Z]\w+ [A-Z]\w+)', sentence)
    proper_nouns_before = [pn4 if pn4!='' else pn3 if pn3!='' else pn2 for pn4, pn3, pn2 in proper_noun_tuples]
    proper_nouns_after = [pn.replace(" ", "_") for pn in proper_nouns_before]
    del proper_noun_tuples

    for i, pn in enumerate(proper_nouns_before):
        sentence = sentence.replace(pn, proper_nouns_after[i])

#     print(sentence, proper_nouns_after)

    return sentence

def sm_tokenize(text, lower=True, remove_stopwords=False, remove_more=[]):
    """
    Tokenize by word level.
    Lower
    Remove stopwords
    How to use:
    converted_sent = sm_tokenize(sent, remove_stopwords=True)
    converted_sent = sm_tokenize(sent, remove_stopwords=True, remove_more=[r'\\n'])

    e.g.,
    INPUT : 5.0 great_location , short walk from amtrak station , link or to a game . price perfect for downtown seattle .
    OUTPUT: 5.0 great_location , short walk amtrak station , link game . price perfect downtown seattle .
    """
    import re
    from nltk.tokenize import word_tokenize # Seal data, already tokenized in word level
    from nltk.corpus import stopwords
    # from nltk.stem.wordnet import WordNetLemmatizer <- useless for text classification
    stopwords = set(stopwords.words('english'))

    words = word_tokenize(text)
    if remove_stopwords : words = [word for word in words if word not in stopwords]

    text = " ".join(words)
    if lower : text = text.lower()
    if remove_more :
        for pattern in remove_more:
            text = re.sub(pattern, '', text)

    return text

def sm_filtering(text, level=1):
    """
    How to use:
    sentence = sm_filtering(text)

    Level0 : Double or more space to one space : '[ ]+' -> '[ ]'
    Level1 : simple filter, Remove not words '\W+'
    Level2 : Convert number to number symbol. e.g., {19231, 14,230, 13.24} -> <num>
    Level3 : Convert number to date symbol.   e.g., {1993.2.12, 1994-2-12, ...} -> <date>
    Level4 : Convert mix number and characters to symbol. e.g., 1923-abc -> <mix_num>
    Level5 : Convert place to symbol : new_york -> <city>, sweden -> <country>
    Level6 : Convert money to symbol : {$ 3 billion, $ 5M, 5 dollars, ...} -> <money>

    """
    # Substring by regular expression, order is important.
    # contract_pattern = '\©\°\♦\□\■\•'
    # 13th, Feb
    SIMPLE_FILTER = r'\b\W+\b'
    REMOVE_PATTERN = r'[\t\:\;\,\.\"\”\“\)\(\[\]\/\\\ˆ\^\<\>]+'
    PURE_NUM_PATTERN = r'(\s+\d+\s+)'
    MIXED_NUM_PATTERN = r'(\w*[\:\;\_\`\.\,\-\*\'\"\/\\&\^\%]*\d+\w*[\:\;\_\`\.\,\-\*\'\"\/\\&\^\%]*)+'
    # ROMAN_NUM = r'\b(ii|iii|iix)\b'

    # 9th, Feb, 2018 : Made filter for pure numbers, mixed numbers in order to unify preprocessing part with Aron.
    # PURE_NUM_PATTERN = r"([ˆ ]+[ ]*\d+ )"                 # Second, remove pure number
    # MIXED_NUM_PATTERN = r'[\w,.\=_\-;:\\\/\"\'\|]*\d+[\w,.\=_\-;:\\\/\"\'\|]*'

    SEPERATE_AST = r'\w+\'\w+'
    TRASH_PATTERN = r'[\?!@#$%ˆ&*()\_\-\|~±§\}\{\+\¬]+'
    FIRST_ONE_CHAR = r'\b[b-hj-z][\.\,]* \b' # Space is important, excluding [a, i]

    #     sent = re.sub(REMOVE_PATTERN, '', line)
    #     sent = re.sub(SEPERATE_AST, ' \' ', sent)
    #     sent = re.sub(TRASH_PATTERN, ' ', sent)
    #     sent = re.sub(FIRST_ONE_CHAR, '', sent)
    sent = re.sub(SIMPLE_FILTER, ' ', line)
    sent = re.sub(r' [ ]+', ' ', sent) # double space -> a space

    result = re.sub(PURE_NUM_PATTERN, ' <NUM> ', test)
    replace_numb = re.match(r'\d+', test)
    result = result.replace(replace_numb.group(), '<NUM>')
    result = re.sub(MIXED_NUM_PATTERN, ' <MIXED> ', result)
    result
    return None


def split_labeled_data(file_path, num_piece=5):
    """
    Split data to pieces, if you want to shuffle, just add shuffled
    How to use:
    split_labeled_data("ag_shuffled_train_fastText.txt")

    """
    percent = (1/num_piece) # 0.2 -> Our taget : 0.2, 0.4, 0.6, 0.8, 1

    total_line = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            total_line+=1

    for i in range(num_piece):
        index = int(total_line*percent*(i+1))
        amount_percent = int(percent*(i+1)*100) # 0.2 -> 20
        write_path = file_path.replace('.txt', '_'+str(amount_percent)+'.txt')
        f_write    = open(write_path, 'w', encoding='utf8')
        with open(file_path, 'r', encoding='utf8') as f_read:
            for i, line in enumerate(f_read):
                if i>= index: break
                if line.endswith('\n'): f_write.write(line)
                else : f_write.write(line+' \n')
        f_write.close()
    print("Successfully made split dataset from ", file_path)

    return None



def build_Language_Model_data(file_path, vocab_size=50000, annotated=True, unk_symbol="<unk>", eos_symbol=False, premade_voca=None):
    """
    How to use: build_Language_Model_data('clean_reviews_1.5M.txt', vocab_size = 50000, annotated=True)
    """
    #============== part1, build vocabulary =============
    if premade_voca:
        vocabulary = premade_voca
    else:
        voca_dictionary = {}
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                words = line.split()[1:]
                for word in words:
                    try : voca_dictionary[word]+=1
                    except : voca_dictionary[word] = 0
        vocabulary = list(voca_dictionary.items())
        print("Total voca size : ", len(vocabulary))
        vocabulary.sort(key=lambda x:x[1], reverse=True)
    #============== Select most common voca =============

    if premade_voca is None:
        vocabulary = vocabulary[:vocab_size]
        vocabulary = {voca:True for voca, freq in vocabulary}
    elif premade_voca:
        vocabulary = {voca:True for voca in premade_voca}

    voca_only_words = list(vocabulary.keys())

    print('============Built vocabulary with {} size ============='.format(vocab_size))

    f_write = open(file_path.replace('.txt', '_LM.txt'), 'w', encoding='utf8')
    try:
        with open(file_path, 'r', encoding='utf8') as f_read:
            for line in f_read:
                words = line.split()

                # Remove labels
                if annotated:
                    words = words[1:]

                # Replace not common words to unknown symbol <unk>
                for i, word in enumerate(words):
                    try : vocabulary[word]     # <--- faster way to use instead of [if word not in vocabulary:]
                    except: words[i] = unk_symbol

                # add end of sentence symbol <eos>, if user want
                if eos_symbol: words.append("<eos>")

                # make words to one line
                line = ' '.join(words)

                # Check if there is new line symbols
                if line.endswith('\n'): f_write.write(line)
                else : f_write.write(line+'\n')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    finally : f_write.close()

    print("Successfully parsed")
    print("Returning vocabulary")
    return voca_only_words

def check_sample_lines(file_path, num_lines=10):
    """
    How to use: check_sample_lines('LM_clean_reviews_1.5M.txt')
    """
    i = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            print(line)
            i+=1
            if i==num_lines: break
    return None

def make_sample_data(file_path, num_lines=2000):
    """
    How to use: make_sample_data('LM_clean_reviews_1.5M.txt')
    """
    i = 0
    f_write = open(file_path+'_'+str(num_lines), 'r', encoding='utf8')
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            f_write.write(line)
            i+=1
            if i==num_lines: break
    return None


def shuffle_text_file(file_path, complexity=50):
    import numpy as np

    total_lines = 0
    texts = list()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            total_lines+=1
            texts.append(line)
    indice = np.arange(total_lines)

    for i in range(complexity):
        np.random.shuffle(indice)

    try:
        if file_path.endswith('.txt') : file_path = file_path[:-4]
        f_write = open(file_path+'_shuffled.txt', 'w', encoding='utf8')

        for idx in indice:
            f_write.write(texts[idx])
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    finally : f_write.close()

    print("Successfully shuffled")
    return None


def convert_to_fastText(file_path, prefix='__label__'):
    file_path_fasttext = file_path.replace('.txt', '_fastText.txt')
    print("Creating new fastText format file : ", file_path_fasttext)
    f_write = open(file_path_fasttext, 'w', encoding='utf8')
    with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                words = line.split()
                label = prefix + words[0]
                sentence = ' '.join(words[1:])
                line = label + ' ' + sentence + '\n'
                f_write.write(line)
    f_write.close()

    print("Successfully converted to fastText format")
    return None


def extract_lines(dict_data, category, remove_stopwords=True, lower_case=True, label_prefix=False):
    """
    How to use:
    lines = extract_lines(json_data, category)
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOP_WORD_REMOVE = remove_stopwords
    LOWER_CASE = lower_case
    FASTTEXT = fasttext
    stopwords = set(stopwords.words('english'))
    lines = list()

    for review in dict_data['Reviews']:
        title   = review['Title']
#         rating  = review['Overall']
        content = review['Content']
        try : content = re.sub('\W+', ' ', content)
        except :
            print("Error occur for substring - Check content : ", content)
            return None
        content =  content +' '+ title

        if STOP_WORD_REMOVE:
            words = word_tokenize(content)
            content = ' '.join([word for word in words if word not in stopwords])

        if LOWER_CASE:
            content = content.lower()

        if label_prefix : line = str(label_prefix) + category.upper() + ' ' + content # fastText-prefix : '__label__'
        else : line = category.upper() + ' ' + content

        lines.append(line)

    return lines



# 13th, March, make split train, test set
def split_train_test_val(file_path, language_model=False, val=0.2):
    import numpy as np
    import random

    total_len = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            total_len+=1

    split_point = int(total_len*val)


    f_train = open(file_path.replace('.txt', '_train.txt'), 'w', encoding='utf8')
    f_test  = open(file_path.replace('.txt', '_test.txt'), 'w', encoding='utf8')
    f_val   = open(file_path.replace('.txt', '_valid.txt'), 'w', encoding='utf8')


    with open(file_path, 'r', encoding='utf8') as f:
        for current_position, line in enumerate(f):
            if language_model:
                if  random.random() <= val:
                    if random.random() <= 0.5 : f_test.write(line) # half goes to test, half goes to valid
                    else : f_val.write(line)

                else : f_train.write(line)

            elif not language_model:
                if current_position < split_point:
                    f_test.write(line)
                elif current_position >= split_point:
                    f_train.write(line)

            # select validation sample by 20%
#             if  random.random() <= val:
#                 f_val.write(line)


    f_train.close()
    f_test.close()
    f_val.close()

    return None


def convert2fastText(file_path, label_prefix=False):
    f_rewrite = open(file_path.replace('.txt', '_fastText.txt'), 'w', encoding='utf8')
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            f_rewrite.write(line)

    f_rewrite.close()
    print("Successfully converted to fastText format : {}".format(file_path))
    return None




# 5th, March, make split train, test set

def split_train_test_val(file_path, language_model=False, val=0.2):
    import numpy as np
    import random

    total_len = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            total_len+=1

    split_point = int(total_len*val)


    f_train = open(file_path.replace('.txt', '_train.txt'), 'w', encoding='utf8')
    f_test  = open(file_path.replace('.txt', '_test.txt'), 'w', encoding='utf8')
    f_val   = open(file_path.replace('.txt', '_valid.txt'), 'w', encoding='utf8')


    with open(file_path, 'r', encoding='utf8') as f:
        for current_position, line in enumerate(f):
            if language_model:
                if  random.random() <= val:
                    f_test.write(line)
                f_train.write(line) # <--------------- train contains all text.

            if not language_model:
                if current_position < split_point:
                    f_test.write(line)
                elif current_position >= split_point:
                    f_train.write(line)

            # select validation sample by 20%
            if  random.random() <= val:
                f_val.write(line)


    f_train.close()
    f_test.close()
    f_val.close()



def nltk_ssl_problem_solver(target='stopwords'):
    """
    Solution of SSL problem to use NLTK download.
    How to use:
        nltk_ssl_problem_solver(target='stopwords')
        nltk_ssl_problem_solver(target='WordNetLemmatizer')
    """
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download(target)

    return None



def report_matrix(target_file_path, description, title, y_true, y_pred, label_dict=False, append=False, torch=False):
    """
    How to use:
    title = description = "lr_"+str(learning_rate)+"_adam_"+str(False)+"_nhid_"+str(nhid_linear)
    report_matrix("./log_multi_models/", description, title, y_true, y_pred, torch=True)
    """
    import datetime, os
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    if torch:
        y_true = np.concatenate([y.cpu().numpy() for y in y_true]).ravel()
        y_pred = np.concatenate([y.cpu().numpy() for y in y_pred]).ravel()

    if label_dict:
        y_true = [label_dict['index2label'][index] for index in y_true]
        y_pred = [label_dict['index2label'][index] for index in y_pred]
    
    
    txt1 = '\n\n\nDescription : '+ str(description) + '\n'
    txt2 = "*"*30 + title +"*"*30 + '\n'
    txt3 = str(metrics.classification_report(y_true, y_pred)) + '\n'
    txt4 = "Accuracy : "+ str(accuracy_score(y_true, y_pred))

    if append:
        target_file_path = target_file_path+'/'+str(title)
        if os.path.isfile(target_file_path):
            with open(target_file_path, 'a', encoding='utf8') as f:
                f.write(txt1+txt2+txt3+txt4)
            print("Save file at ", target_file_path)
            return print(txt1, txt2, txt3, txt4)

    if not append :
        target_file_path += str(description) + datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M.txt")

    print("Save file at ", target_file_path)
    with open(target_file_path, 'w', encoding='utf8') as f:
        f.write(txt1+txt2+txt3+txt4)

    return print(txt1, txt2, txt3, txt4)




def make_labeled_file_per_class(input_file, output_file, instance_per_class=0, shuffle=True, complexity=20, label_position=0):
	"""
	How to use :
	file_path = "/Users/sungmin/Desktop/transfer_learning/Dataset/Seal-Software/41k_downsampled/2k_sample_provisions.txt"
	trainloader = get_train_loader(file_path)
	for
	"""
	import numpy as np
    
	label_dict = dict()
	traindata  = list()
    
	with open(input_file, 'r', encoding='utf8') as f:
		for line in f:
			label = line.split()[label_position]
			try : traindata_dict[label] += 1
			except : traindata_dict[label] == 1

			if traindata_dict[label] >= instance_per_class:
				continue

			traindata.append(line.rstrip('\n'))

	if shuffle:
		for _ in range(complexity):
			np.random.shuffle(traindata)


	with open(output_file, 'w', encoding='utf8') as f:
		for line in traindata:
			if not line.endswith('\n') : line+=' \n'
			f.write(line)

	return print("Successfully make labeled file per class, per class {} instance exist".format(instance_per_class))




def file_read_MultiProcess(file_path, process_func=False, num_workers=1, batch_size=4):
	import multiprocessing as mp

	if not process_func:
		print("You did not provide process function, using defualt function which returns just line")
		def process_fun(inputs):
			return inputs

	cpu_num = mp.cpu_count()
	print("You have {} CPU avaiable. You chose {} CPU for processing".format(cpu_num, num_workers))

	pool = mp.Pool(num_workers)
	with open(file_path, 'r', encoding='utf8') as source_file:
		results = pool.map(process_line, source_file, batch_size)
	pool.close()

	return result

