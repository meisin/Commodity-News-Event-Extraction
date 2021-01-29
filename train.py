
import argparse
import logging
import numpy as np
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
import os
import logging
import pandas as pd
from datetime import datetime
from torch.utils import data

from data.const import MODEL_CLASSES
from data.data_loader import CommodityDataset, pad
from model.event_extraction import EventExtractor
from utils.parse_tree import Tree, head_to_tree, tree_to_adj, tokens_tree_adjmatrix
from utils.helper_functions import get_positions, preprocessing, calculate_metric, get_act_sent_len, sent_padding, find_triggers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

from data.data_loader import all_triggers_tag, trigger_tag2idx, idx2trigger_tag, \
                             all_entities, entity2idx, idx2entity, \
                             all_postags, postag2idx, idx2postag, \
                             all_arguments, argument2idx, idx2argument, \
                             all_arguments_tag, argument_tag2idx, idx2argument_tag, \
                             all_modality, modality2idx, idx2modality, \
                             all_polarity, polarity2idx, idx2polarity, \
                             all_deptags, deptags2idx, idx2deptags 


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
    
tb_writer = SummaryWriter() 
   
datetime_object = str(datetime.now())
date = datetime_object.split(' ')[0]
time = datetime_object.split(' ')[1].replace(":", "_")

filename = 'runs/logfiles/output_' + date + time + '.log'

### Setup logging
logging.basicConfig(level=logging.INFO, filename=filename, filemode='w')
logger = logging.getLogger(__name__)

def train_classifyEvent(model, train_dataset, optimizer, criterion_trigger, criterion_argument, epoch, co_train, device, args):
    """ Training """
    tr_loss = 0
    model.train()
    total_len = len(train_dataset)
    for i, batch in enumerate(train_dataset):
        words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
        batch_copy = copy.deepcopy(batch)
        
        tokens_x_2d, entities_x_3d, postags_x_2d, entity_mention_x_2d, triggers_y_2d, argument_tags_2d, arguments_2d, \
             seqlens_1d, head_indexes_2d, words_2d, adjmatrix, event_polarity, event_modality, deptags, \
             gov_words, attention_mask = batch_copy
        
        sent_len_list = get_act_sent_len(head_indexes_2d)
        
        words_in_path = preprocessing(tokens_x_2d, head_indexes_2d)
        selected_pos = preprocessing(postags_x_2d, head_indexes_2d)
        selected_entity = preprocessing(entity_mention_x_2d, head_indexes_2d)    
        triggers_y_2d = preprocessing(triggers_y_2d, head_indexes_2d)
                
        # Move input to GPU
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(device)      
        selected_pos = torch.LongTensor(selected_pos).to(device)
        selected_entity = torch.LongTensor(selected_entity).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)  
        
        batch_size = triggers_y_2d.shape[0] 
                
        optimizer.zero_grad()
                   
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(device)    
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(device)
        encoded_tokens = model.encode_tokens(tokens_x_2d, attention_mask)  
        
        selected_tokens = [[0]] * batch_size
    
        for i in range(batch_size):
            selected_tokens[i] = torch.index_select(encoded_tokens[i], 0, head_indexes_2d[i])

        selected_tokens = torch.stack(selected_tokens)
        
        ### trigger-event classification
        event_type_hat_logits, event_type_hat_2d = model.classify_event(selected_tokens, selected_pos, selected_entity)            
        event_type_hat_logits = event_type_hat_logits.view(-1, event_type_hat_logits.shape[-1])
                       
        trigger_loss = criterion_trigger(event_type_hat_logits, triggers_y_2d.view(-1))  
        
        ####   argument linking ######
        trigger_entity_hidden, trigger_entity_keys = \
                model.build_trigger_entity(encoded_tokens, selected_pos, selected_entity, event_type_hat_2d, arguments_2d)
           
        if len(trigger_entity_keys) > 0:  
            maxlen = max(sent_len_list)            
            pair_len = []
            words_in_path_all = []
            selected_sent = []
            selected_gov_list = []
            trigger_position = []
            entity_position = []

            for pair in trigger_entity_keys:            
                (pair_index, t_start, t_end, t_type_str, e_start, e_end, e_type_str) = pair
                pair_len.append(sent_len_list[pair_index])

                selected_sent.append(selected_tokens[pair_index][0:maxlen])
                        
                gov_words_tensor = torch.LongTensor(sent_padding(gov_words[pair_index], maxlen)).to(device)                    
                selected_gov_list.append(gov_words_tensor)

                trigger_position_tensor = torch.LongTensor(sent_padding(get_positions(t_start, (t_end-1), sent_len_list[pair_index]), maxlen))
                trigger_position.append(trigger_position_tensor)

                entity_position_tensor = torch.LongTensor(sent_padding(get_positions(e_start, (e_end-1), sent_len_list[pair_index]), maxlen))
                entity_position.append(entity_position_tensor)
                words_in_path_all.append(words_in_path[pair_index])

            selected_sent = torch.stack(selected_sent)  
            selected_gov_list = torch.stack(selected_gov_list)
            trigger_position = torch.stack(trigger_position)
            entity_position = torch.stack(entity_position)
            
            l = np.array(pair_len)              
            adj = tokens_tree_adjmatrix(words_in_path_all, selected_gov_list, selected_sent, l, args.DIST, trigger_position, entity_position, maxlen)

            trigger_mask, entity_mask = trigger_position.eq(0).eq(0).unsqueeze(2), entity_position.eq(0).eq(0).unsqueeze(2) # invert mask
            trigger_mask = torch.BoolTensor(trigger_mask).to(device)
            entity_mask = torch.BoolTensor(entity_mask).to(device)              
            
            trigger_entity_hat_logits, pooling_output, trigger_entity_y_1d, roles_hat = \
                    model.classify_argument_roles_gcn(selected_tokens, selected_pos, selected_entity, adj, trigger_mask, \
                                                     entity_mask, trigger_entity_keys, maxlen, arguments_2d)
            
            trigger_entity_y_1d = torch.LongTensor(trigger_entity_y_1d).to(device)
            trigger_entity_loss = criterion_argument(trigger_entity_hat_logits, trigger_entity_y_1d)  

            loss = trigger_loss + 2 * trigger_entity_loss
        else:
            loss = trigger_loss    
        
        
        tr_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    return (tr_loss / len(train_dataset))
        
def eval_classifyEvent(model, dev_dataset, optimizer, criterion_trigger, criterion_argument, epoch, co_train, device, args):
    """ Evaluation """
    eval_loss = 0
    model.eval()
    triggers_y_2d_all, event_type_hat_2d_all = [[]], [[]]
    trigger_entity_hat_1d_all, trigger_entity_y_1d_all, roles_hat_all = [], [], []
    result = {}
    
    with torch.no_grad():
        for i, batch in enumerate(dev_dataset):
            words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
            batch_copy = copy.deepcopy(batch)
            
            tokens_x_2d, entities_x_3d, postags_x_2d, entity_mention_x_2d, triggers_y_2d, argument_tags_2d, arguments_2d, \
                seqlens_1d, head_indexes_2d, words_2d, adjmatrix, event_polarity, event_modality, deptags, \
                gov_words, attention_mask = batch_copy
                    
            sent_len_list = get_act_sent_len(head_indexes_2d)
        
            words_in_path = preprocessing(tokens_x_2d, head_indexes_2d)
            selected_pos = preprocessing(postags_x_2d, head_indexes_2d)
            selected_entity = preprocessing(entity_mention_x_2d, head_indexes_2d)    
            triggers_y_2d = preprocessing(triggers_y_2d, head_indexes_2d)            

            # Move input to GPU
            triggers_y_2d = torch.LongTensor(triggers_y_2d).to(device)      
            selected_pos = torch.LongTensor(selected_pos).to(device)
            selected_entity = torch.LongTensor(selected_entity).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device)  
            
            tokens_x_2d = torch.LongTensor(tokens_x_2d).to(device)    
            head_indexes_2d = torch.LongTensor(head_indexes_2d).to(device)
            encoded_tokens = model.encode_tokens(tokens_x_2d, attention_mask)   
            
            batch_size = triggers_y_2d.shape[0] 
            selected_tokens = [[0]] * batch_size
    
            for i in range(batch_size):
                selected_tokens[i] = torch.index_select(encoded_tokens[i], 0, head_indexes_2d[i])
                    
            selected_tokens = torch.stack(selected_tokens)
            #### trigger-event classification            
            event_type_hat_logits, event_type_hat_2d = model.classify_event(selected_tokens, selected_pos, selected_entity)                        
            event_type_hat_logits = event_type_hat_logits.view(-1, event_type_hat_logits.shape[-1])                                

            trigger_loss = criterion_trigger(event_type_hat_logits, triggers_y_2d.view(-1)) 
            
            ####   evaluation for argument linking ######
            trigger_entity_hidden, trigger_entity_keys = \
                    model.build_trigger_entity(encoded_tokens, selected_pos, selected_entity, event_type_hat_2d, arguments_2d)

            if len(trigger_entity_keys) > 0:
                maxlen = max(sent_len_list)            
                pair_len = []
                words_in_path_all = []
                selected_sent = []
                selected_gov_list = []
                trigger_position = []
                entity_position = []

                for pair in trigger_entity_keys:            
                    (pair_index, t_start, t_end, t_type_str, e_start, e_end, e_type_str) = pair
                    pair_len.append(sent_len_list[pair_index])

                    selected_sent.append(selected_tokens[pair_index][0:maxlen])

                    gov_words_tensor = torch.LongTensor(sent_padding(gov_words[pair_index], maxlen)).to(device)                    
                    selected_gov_list.append(gov_words_tensor)

                    trigger_position_tensor = torch.LongTensor(sent_padding(get_positions(t_start, (t_end-1), sent_len_list[pair_index]), maxlen))
                    trigger_position.append(trigger_position_tensor)

                    entity_position_tensor = torch.LongTensor(sent_padding(get_positions(e_start, (e_end-1), sent_len_list[pair_index]), maxlen))
                    entity_position.append(entity_position_tensor)
                    words_in_path_all.append(words_in_path[pair_index])

                selected_sent = torch.stack(selected_sent)  
                selected_gov_list = torch.stack(selected_gov_list)
                trigger_position = torch.stack(trigger_position)
                entity_position = torch.stack(entity_position)

                l = np.array(pair_len)              
                adj = tokens_tree_adjmatrix(words_in_path_all, selected_gov_list, selected_sent, l, args.DIST, trigger_position, entity_position, maxlen)

                trigger_mask, entity_mask = trigger_position.eq(0).eq(0).unsqueeze(2), entity_position.eq(0).eq(0).unsqueeze(2) # invert mask
                trigger_mask = torch.BoolTensor(trigger_mask).to(device)
                entity_mask = torch.BoolTensor(entity_mask).to(device)              

                trigger_entity_hat_logits, pooling_output, trigger_entity_y_1d, roles_hat = \
                        model.classify_argument_roles_gcn(selected_tokens, selected_pos, selected_entity, adj, trigger_mask, \
                                                         entity_mask, trigger_entity_keys, maxlen, arguments_2d)       
                
                trigger_entity_y_1d = torch.LongTensor(trigger_entity_y_1d).to(device)
                trigger_entity_loss = criterion_argument(trigger_entity_hat_logits, trigger_entity_y_1d)

                roles_hat = roles_hat.cpu().numpy().tolist()
                trigger_entity_y_1d = trigger_entity_y_1d.cpu().numpy().tolist()

                #print([idx2argument[n] for n in trigger_entity_y_1d]) 
                #print('------')
                #print([idx2argument[n] for n in roles_hat])   
                
                roles_hat_all = roles_hat_all + roles_hat
                trigger_entity_y_1d_all = trigger_entity_y_1d_all + trigger_entity_y_1d             
    

                loss = trigger_loss + 2 * trigger_entity_loss
            else:
                loss = trigger_loss    
            
            eval_loss += loss.item()
              
            triggers_y_2d_all = triggers_y_2d_all + triggers_y_2d.cpu().numpy().tolist()  
            event_type_hat_2d_all = event_type_hat_2d_all + event_type_hat_2d.cpu().numpy().tolist()
            
    accuracy, precision, recall, f1, report = calculate_metric(triggers_y_2d_all, event_type_hat_2d_all, toPrint= False, \
                                                       classify_type='event', labels=idx2trigger_tag)
       
    result['trg_accuracy'] = accuracy
    result['trg_precision'] = precision
    result['trg_recall'] = recall
    result['trg_f1'] = f1
    result['trg_report'] = report
    
    truth_all_array = np.asarray(trigger_entity_y_1d_all)
    prediction_all_array = np.asarray(roles_hat_all)
            
    arg_accuracy = accuracy_score(truth_all_array, prediction_all_array)
    arg_precision = precision_score(truth_all_array, prediction_all_array, average="weighted", zero_division='warn')
    arg_recall = recall_score(truth_all_array, prediction_all_array, average="weighted",zero_division='warn')
    arg_f1 = f1_score(truth_all_array, prediction_all_array, average="weighted",zero_division='warn')
    
    try:   
        arg_classification_report = classification_report(truth_all_array, prediction_all_array)
        result['arg_accuracy'] = arg_accuracy
        result['arg_precision'] = arg_precision
        result['arg_recall'] = arg_recall
        result['arg_f1'] = arg_f1
        result['arg_report'] = arg_classification_report
    except Exception as ex:
        result['arg_accuracy'] = 0
        result['arg_precision'] = 0
        result['arg_recall'] = 0
        result['arg_f1'] = 0
        result['arg_report'] = ''
    
    return (eval_loss / len(dev_dataset)), result
    

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True, help="The input training data file (a json file).")
    
    parser.add_argument("--eval_data_file", default=None, type=str, required=True, help="The input testing data file (a json file).")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU for training.")
    parser.add_argument("--transformer_type", default="bert", type=str, help="Available options are: bert, roberta and combert") 
    parser.add_argument("--learning_rate", default= 0.00002, type=float, help="The learning rate for model's optimizer.")
    parser.add_argument("--n_epochs", default=1, type=int, help="Number of training epochs.")
    
    ## Multi-channel embeddings
    parser.add_argument("--postag_embedding_dim", default=50, type=int, help="POS tag embedding dimension")
    parser.add_argument("--deptags_embedding_dim", default=50, type=int, help="UD Dependency tag embedding dimension")
    parser.add_argument("--trigger_embedding_dim", default=50, type=int, help="trigger embedding dimension")
    parser.add_argument("--entity_embedding_dim", default=50, type=int, help="Entity tag embedding dimension")
    
    ## for GCN layers
    parser.add_argument("--GCN_hidden_dim", default=200, type=int, help="GCN input dimension")
    parser.add_argument("--GCN_num_layers", default=2, type=int, help="Number of GCN layers")   
    parser.add_argument("--DIST", default=1, type=int, help="Build contextual sub-tree that is DIST away from shortest dependency path")
    
    ## for Modality and Polarity Classification which uses a Bi-LSTM
    parser.add_argument("--bidirectional", default=True, type=bool, help="Whether the Bi-LSTM is bidirectional or not")
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers in Bi-LSTM")
    parser.add_argument("--dropout", default=0.25, type=float, help="dropout rate")


    args = parser.parse_args()
    args.transformer_type       
    
    Tokenizer = MODEL_CLASSES[args.transformer_type][2]
    model_name_or_path = MODEL_CLASSES[args.transformer_type][3]
    tokenizer = Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False, cache_dir = None)
   
    logger.info("Training/evaluation parameters %s", args)
          
        
    ### Define Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_classify = EventExtractor(args, event_size = len(all_triggers_tag), \
                                    argument_size = len(all_arguments), argument_tags_size = len(all_arguments_tag),
                                    entity_size = len(all_entities), postag_size = len(all_postags), \
                                    deptags_size = len(all_deptags), trigger_size = len(all_triggers_tag))

    model_classify = model_classify.to(device)

    optimizer = optim.Adam(model_classify.parameters(), lr = args.learning_rate)    ## use back the same optimizer
    #optimizerSGD = optim.SGD(model_classify.parameters(), lr=learning_rate, momentum=0.9)
    criterion_classify = nn.CrossEntropyLoss(ignore_index = 0) #nn.BCEWithLogitsLoss()    
    criterion_withoutPadding = nn.CrossEntropyLoss()
    
    
    ### training
    trainset = args.train_data_file
    train_dataset = CommodityDataset(trainset, tokenizer, args.transformer_type)
    train_dataset = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, collate_fn = pad)
    
            
    ### Evaluation    
    testset = args.eval_data_file
    test_dataset = CommodityDataset(testset, tokenizer, args.transformer_type)
    test_dataset = data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, collate_fn = pad)
    
    global_step = 0
    co_train = True
    logger.info("***** Running training *****")
    for epoch in range(1, args.n_epochs + 1):   
        global_step += 1
        print('epoch # ' + str(epoch))
        tr_loss = train_classifyEvent(model_classify, train_dataset, optimizer, criterion_classify, criterion_withoutPadding, epoch, co_train, device, args)
        print('training loss : ' + str(tr_loss))
        eval_loss, result = eval_classifyEvent(model_classify, test_dataset, optimizer, criterion_classify, criterion_withoutPadding, epoch, co_train, device, args)
        print('evaluation loss : ' + str(eval_loss))    
        
        print('=====Trigger Classification=====')
        print('accuracy - ' + str(result['trg_accuracy']))
        print('precision - ' + str(result['trg_precision']))
        print('recall - ' + str(result['trg_recall']))
        print('f1 - ' + str(result['trg_f1']))
        print('classification report - ' + str(result['trg_report']))
        print('=======Argument Role Classification=========')   
        print('arg accuracy - ' + str(result['arg_accuracy']))
        print('arg precision - ' + str(result['arg_precision']))
        print('arg recall - ' + str(result['arg_recall']))
        print('arg f1 - ' + str(result['arg_f1']))
        print('arg classification report - ' + str(result['arg_report']))
       
        
        ## Write to logging file
        logger.info('epoch # ' + str(epoch))
        logger.info('=====Trigger Classification=====')
        logger.info('accuracy - ' + str(result['trg_accuracy']))
        logger.info('precision - ' + str(result['trg_precision']))
        logger.info('recall - ' + str(result['trg_recall']))
        logger.info('f1 - ' + str(result['trg_f1']))
        logger.info(result['trg_report'])              
        logger.info('=======Argument Role Classification=========')   
        logger.info('arg accuracy - ' + str(result['arg_accuracy']))
        logger.info('arg precision - ' + str(result['arg_precision']))
        logger.info('arg recall - ' + str(result['arg_recall']))
        logger.info('arg f1 - ' + str(result['arg_f1']))   
        logger.info(result['arg_report'])
        
        ## Write results to TensorBoard
        tb_writer.add_scalar("training_loss", tr_loss, global_step)     
        tb_writer.add_scalar("eval_loss", eval_loss, global_step)     
        tb_writer.add_scalar("precision", result['trg_precision'], global_step)
        tb_writer.add_scalar("recall", result['trg_recall'], global_step)
        tb_writer.add_scalar("accuracy", result['trg_accuracy'], global_step)
        tb_writer.add_scalar("f1", result['trg_f1'], global_step) 
        tb_writer.add_scalar("argument_accuracy", result['arg_accuracy'], global_step)
        tb_writer.add_scalar("argument_precision", result['arg_precision'], global_step)
        tb_writer.add_scalar("argument_recall", result['arg_recall'], global_step)
        tb_writer.add_scalar("argument_f1", result['arg_f1'], global_step)     

if __name__ == "__main__":
    main()