import numpy as np

from data.const import NONE

def build_vocab(labels, PAD, BIO_tagging=True, padding=True):
    """ build vocabulary for labels """
    if padding:
        all_labels = [PAD, NONE]
    else:
        all_labels = []
        
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    
    return all_labels, label2idx, idx2label
    
def get_positions(start_idx, end_idx, length):
    """ Get trigger/argument position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))
            
            
def preprocessing(list_of_tokens, head_indexes_2d):
    """ Get head indexes only for word-pieces tokenization  """
    batch_size = len(list_of_tokens)
        
    xx = [[]] * batch_size
        
    for i in range(batch_size):
        xx[i] = [0] * len(list_of_tokens[i])
        
        for idx, item in enumerate(head_indexes_2d[i]):
            if item == 0:
                xx[i][idx] = 0
            else:
                xx[i][idx] = list_of_tokens[i][head_indexes_2d[i][idx]]
        
    return xx  
 
def calculate_metric(y, y_, toPrint, classify_type, labels):
    """ calculate precision, recall, accuracy and F1 of prediction """
    from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
    
    y_filtered = [[] for _ in range(len(y))]
    y__filtered = [[] for _ in range(len(y_))]
    
       
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j] != 0:
                y_filtered[i].append(labels[y[i][j]])
                y__filtered[i].append(labels[y_[i][j]])               
                   
    if toPrint is True:
        print(y_filtered)
        print('---')
        print(y__filtered)

    report = classification_report(y_filtered, y__filtered)
        
    return accuracy_score(y_filtered, y__filtered), precision_score(y_filtered, y__filtered), recall_score(y_filtered, y__filtered), \
        f1_score(y_filtered, y__filtered), report

def get_act_sent_len(head_indexes_2d):
    """ Get actual sentence length (counting only head indexes)  """
    sent_len_list = [0] * len(head_indexes_2d)
    
    for idx, head_list in enumerate(head_indexes_2d):
        sent_len = 0
        for item in head_list:
            if item != 0:
                sent_len += 1
        sent_len_list[idx] = sent_len
    return sent_len_list
    
def sent_padding(tokens_list, SEQ_LEN):
    """ Padding the token list with '-1' up to SEQ_LEN  """
    final_list = tokens_list      
    
    for i in range(len(tokens_list), SEQ_LEN):
        final_list.append(-1)
    return final_list
    
def find_triggers(labels):
    """
    : param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    : return: [(0, 2, 'Conflict Attack'), (3, 4, 'Life Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]
    
    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i+1, "-".join(labels[i][1:])])
    
    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break
    
    return [tuple(item) for item in result]