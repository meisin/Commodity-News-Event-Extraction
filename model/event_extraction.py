"""
Module for building Event Extraction Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from data.const import MODEL_CLASSES, INFINITY_NUMBER
from data.data_loader import all_triggers_tag, trigger_tag2idx, idx2trigger_tag, \
                             all_entities, entity2idx, idx2entity, \
                             all_postags, postag2idx, idx2postag, \
                             all_arguments, argument2idx, idx2argument, \
                             all_arguments_tag, argument_tag2idx, idx2argument_tag, \
                             all_modality, modality2idx, idx2modality, \
                             all_polarity, polarity2idx, idx2polarity, \
                             all_deptags, deptags2idx, idx2deptags 
from utils.helper_functions import find_triggers

from model.graph_convolution import GCN

class EventExtractor(nn.Module):
    """ Event Extractor model """
    def __init__(self, args, event_size, argument_size, argument_tags_size, entity_size, \
                 postag_size, deptags_size, trigger_size):
        
        super().__init__()
              
        config_class, model_class, tokenizer_class, model_name_or_path = MODEL_CLASSES[args.transformer_type]
        
        self.transformer = model_class.from_pretrained(model_name_or_path)
                    
        self.postag_embed = nn.Embedding(num_embeddings = postag_size,
                                         embedding_dim = args.postag_embedding_dim)
        
        self.deptags_embed = nn.Embedding(num_embeddings = deptags_size,
                                         embedding_dim = args.deptags_embedding_dim)
        
        self.trigger_embed = nn.Embedding(num_embeddings = trigger_size,
                                          embedding_dim = args.trigger_embedding_dim)
        
        self.entity_mention_embed = nn.Embedding(num_embeddings = entity_size,
                                                 embedding_dim = args.entity_embedding_dim)
        
        self.model_embedding_dim = 768    ### embedding size for BERT-BASE-CASED
              
        self.model_input_size = self.model_embedding_dim +  args.postag_embedding_dim + args.entity_embedding_dim  
        
               
        # gcn layer - Pruned dependency tree
        self.gcn = GCN(args.GCN_hidden_dim, args.GCN_num_layers, self.model_input_size)
        
        self.out_mlp = nn.Sequential(nn.Linear(args.GCN_hidden_dim*3, args.GCN_hidden_dim), nn.ReLU())
        ## *3 because appending together trigger_ouputs and entity_outputs
        
        self.fc_argument_roles_gcn = nn.Sequential(nn.Linear(args.GCN_hidden_dim, argument_size))
        
        self.fc = nn.Sequential(nn.Linear(self.model_input_size, event_size)) 
        
        self.fc_entity_mention = nn.Sequential(nn.Linear(self.model_embedding_dim, entity_size))
    
    def classify_event(self, encoded_tokens, selected_pos, selected_entity):    
        """ Function to classify Event Trigger (token classification task) """
        postags_embed = self.postag_embed(selected_pos)    
        entity_embed = self.entity_mention_embed(selected_entity)       
        input_tensor = torch.cat([encoded_tokens, postags_embed, entity_embed], 2)
                
        event_hat_logits = self.fc(input_tensor)
        event_hat_2d = event_hat_logits.argmax(-1)        
           
        return event_hat_logits, event_hat_2d   
    
    def encode_tokens(self, selected_tokens, attention_mask):
        """ Function to encode tokens with the selected Pre-trained Language Model """        
        outputs = self.transformer(input_ids = selected_tokens, attention_mask = attention_mask)
        word_enc = outputs[0]      ### the last hidden state
        
        return word_enc
    
    
    def build_trigger_entity(self, encoded_tokens, selected_pos, selected_entity, event_hat_2d, arguments_2d):        
        """ Function to combine trigger with candidate entity """
        postags_embed = self.postag_embed(selected_pos)               
        input_tensor = torch.cat([encoded_tokens, postags_embed], 2)
        
        trigger_embed = self.trigger_embed(event_hat_2d)        
        event_xx = torch.cat([input_tensor, trigger_embed], 2)         
        
        event_window_size = 3
        argument_window_size = 10
                
        entity_embed = self.entity_mention_embed(selected_entity)        
        argument_xx = torch.cat([input_tensor, entity_embed], 2)  
        
        batch_size = input_tensor.shape[0]         
        
        trigger_entity_hidden, trigger_entity_keys = [], []
        golden_argument_all = []        
        
        for i in range(batch_size):               
            predicted_triggers = find_triggers([idx2trigger_tag[trigger] for trigger in event_hat_2d[i].tolist()])
            actual_sent_len = 0
            for t in event_hat_2d[i]:
                if t > 0: 
                    actual_sent_len += 1
            
            if len(predicted_triggers) > 5:   ## prune away number of predicted trigger that is more than 5 
                                              ## assumption: each sentence has less than 5 events
                continue
                
            candidates = arguments_2d[i]['candidates']    
            for candidate in candidates:
                argument_tensor = torch.zeros([argument_window_size, self.model_input_size])
                e_start, e_end, e_type_str = candidate
                j = 0
                
                for idx in range(e_start, e_end):
                    argument_tensor[j] = argument_xx[i, idx, ]
                    j += 1
                golden_argument_all.append(argument_tensor)  
   
            for predicted_trigger in predicted_triggers:      
                
                event_tensor = torch.zeros([event_window_size, 868])
                t_start, t_end, t_type_str = predicted_trigger
                
                if (t_end - t_start) > event_window_size:
                    continue
                    
                k = 0
                for idx in range(t_start, t_end):
                    event_tensor[k] = event_xx[i, idx, ]
                    k += 1
                    
                for c, candidate in enumerate(candidates):
                    e_start, e_end, e_type_str = candidate
                    
                    ### trigger_entity_keys is the permutation of predited_triggers with golden-entity-candidates  
                    trigger_entity_hidden.append(torch.cat([event_tensor, golden_argument_all[c]], 0))
                    trigger_entity_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))                                           
                   
        return trigger_entity_hidden, trigger_entity_keys
        
    
    def classify_argument_roles_gcn(self, selected_tokens, selected_pos, selected_entity, adj, trigger_mask, entity_mask, \
                                    trigger_entity_keys, maxlen, arguments_2d):
        """ Function classify event argument roles """ 
        """ Input: Combination of trigger with candidate golden entity """
        
        postags_embed = self.postag_embed(selected_pos)    
        entity_embed = self.entity_mention_embed(selected_entity)   
        input_tensor = torch.cat([selected_tokens, postags_embed, entity_embed], 2)    ## torch.Size([2, 54, 868])
        input_tensor = input_tensor[:, 0:maxlen, :]        
              
        gcn_inputs_list = []
        for pair in trigger_entity_keys:            
            (pair_index, t_start, t_end, t_type_str, e_start, e_end, e_type_str) = pair
            gcn_inputs_list.append(input_tensor[pair_index])
        
        gcn_inputs_list = torch.stack(gcn_inputs_list)
        
        # gcn layer
        gcn_outputs, pool_mask = self.gcn(adj, gcn_inputs_list) 

        ## Max pooling
        pooling_output = pool(gcn_outputs, pool_mask, type='max') 
        trigger_out = pool(gcn_outputs, trigger_mask, type='max')
        entity_out = pool(gcn_outputs, entity_mask, type='max')        
        
        outputs = torch.cat([pooling_output, trigger_out, entity_out], dim=1) 
                
        outputs = self.out_mlp(outputs)  
        logits = self.fc_argument_roles_gcn(outputs)
        roles_hat = logits.argmax(-1)
        
        ### Get the golden=trigger-entity (ground truth)
        trigger_entity_y_1d = []
        for (i, t_start, t_end, t_type_str, e_start, e_end, e_type_str) in trigger_entity_keys:
            a_label = argument2idx['NONE']
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:   ## if predicted trigger exist in Golden-Events
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            trigger_entity_y_1d.append(a_label)   ## filter out only matched arguments (keep only argument labels)
        
        return logits, pooling_output, trigger_entity_y_1d, roles_hat

def pool(h, mask, type='max'):
    """ Pooling function """ 
    if type == 'max':
        h = h.masked_fill(mask, -INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

