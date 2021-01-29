"""
Functions to import data from json files
"""

import json
from torch.utils import data
from utils.helper_functions import build_vocab
from data.const import transformer_type, NONE, PAD_BERT, CLS_BERT, SEP_BERT, PAD_ROBERTA, CLS_ROBERTA, SEP_ROBERTA, TRIGGERS, ARGUMENTS, NOMINAL_ENTITIES, POSTAGS, NER, ENTITIES, VERB_FAMILY, DEPTAGS, POLARITY, MODALITY

import numpy as np
import copy

PAD = PAD_BERT    ## to-do: need to include also PAD_ROBERTA 

# init vocab
all_triggers_tag, trigger_tag2idx, idx2trigger_tag = build_vocab(TRIGGERS, PAD)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES, PAD)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, PAD, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, PAD, BIO_tagging=False, padding=False)
all_arguments_tag, argument_tag2idx, idx2argument_tag = build_vocab(ARGUMENTS, PAD)
all_modality, modality2idx, idx2modality = build_vocab(MODALITY, PAD, BIO_tagging=False, padding=False)
all_polarity, polarity2idx, idx2polarity = build_vocab(POLARITY, PAD, BIO_tagging=False, padding=False)
all_deptags, deptags2idx, idx2deptags = build_vocab(DEPTAGS, PAD, BIO_tagging=False, padding=True)


class CommodityDataset(data.Dataset):
    """ A Module to read in dataset in the form of .json file """
    def __init__(self, fpath, tokenizer, transformer_type):
        
        if transformer_type == "bert" or transformer_type == "combert":
            # for BERT
            self.PAD = PAD_BERT
            self.CLS = CLS_BERT
            self.SEP = SEP_BERT
        else:
            # for ROBERTA
            self.PAD = PAD_ROBERTA
            self.CLS = CLS_ROBERTA
            self.SEP = SEP_ROBERTA
            
        self.tokenizer = tokenizer
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.argument_tags_li, \
                    self.arguments_li, self.entity_mention_li, self.adjmatrix_li, self.event_polarity_li, \
                    self.event_modality_li, self.deptags_li, self.gov_words_li = \
                            [], [], [], [], [], [], [], [], [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                sentence = item['sentence']
                words = item['words']
                entities = [[NONE] for _ in range(len(words))]
                nominal_entities = [NONE] * len(words)            
                triggers = [NONE] * len(words)
                argument_tags = [NONE] * len(words)      
                postags = item['pos-tags']
                adjmatrix = item["stanford-colcc"]
                event_polarity = []
                event_modality = []      
                
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                for entity_mention in item['golden-entity-mentions']:
                    arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type'].upper()))

                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type'].upper()
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)
                            
                        nominal_entities[i] = entity_type

                        
                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:                    
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        trigger_type = event_mention['event_type'].upper()
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)
                    
                    event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'].upper())
                    
                    ## event modality & polarity
                    event_modality.append((event_mention['trigger']['start'], event_mention['trigger']['end'], \
                                           event_mention['event_type'].upper(), modality2idx[event_mention['modality'].upper()]))
                    
                    event_polarity.append((event_mention['trigger']['start'], event_mention['trigger']['end'], \
                                           event_mention['event_type'].upper(), polarity2idx[event_mention['polarity'].upper()]))
                    
                    ## event arguments
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role.upper()]))
                        for i in range(argument['start'], argument['end']):
                            if i == argument['start']:
                                argument_tags[i] = 'B-{}'.format(role.upper())
                            else:
                                argument_tags[i] = 'I-{}'.format(role.upper())
                

                adjmatrix_, deptags, gov_words = generateAdjMatrix(adjmatrix, len(words))  
                
                NER_list = item['ner']
                ## convert NER_list to bio tags
                for idx, token in enumerate(NER_list):
                    if token != 'O':
                        if idx == 0:   ## first word in the sentence
                            NER_list[idx] = 'B-{}'.format(token)
                        elif idx > 0:
                            if NER_list[idx-1] == 'O':
                                NER_list[idx] = 'B-{}'.format(token)
                            elif NER_list[idx-1] != 'O':
                                NER_list[idx] = 'I-{}'.format(token)
                                
                combined_entity = combine_named_nominal_entities(NER_list, nominal_entities)
                
                if len(combined_entity) == 0:
                    '''problematic sentence'''
                    print(item['sentence_id'])
                    print(len(words))
                    print(words)
                    print(len(NER_list)) 
                    print(NER_list)
                    print(len(nominal_entities))
                    print(nominal_entities)
                                        
                self.sent_li.append([self.CLS] + words + [self.SEP])
                self.entities_li.append([[self.PAD]] + entities + [[self.PAD]])
                self.postags_li.append([self.PAD] + postags + [self.PAD])
                self.triggers_li.append([self.PAD] + triggers + [self.PAD])
                self.argument_tags_li.append([self.PAD] + argument_tags + [self.PAD])   ### combined nominal and Named Entity together
                self.arguments_li.append(arguments)
                self.entity_mention_li.append([self.PAD] + combined_entity + [self.PAD])                
                self.adjmatrix_li.append(adjmatrix_)
                self.event_polarity_li.append(event_polarity)
                self.event_modality_li.append(event_modality)
                self.deptags_li.append(deptags)
                self.gov_words_li.append(gov_words)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, argument_tags, arguments, entity_mention, adjmatrix, event_polarity, \
        event_modality, deptags, gov_words = \
                    self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], \
                    self.argument_tags_li[idx], self.arguments_li[idx], self.entity_mention_li[idx], \
                    self.adjmatrix_li[idx], self.event_polarity_li[idx], self.event_modality_li[idx], self.deptags_li[idx], \
                    self.gov_words_li[idx]
        
      
        tokens_x, entities_x, postags_x, is_heads, entity_mention_x, triggers_y, argument_tags_y = \
                     [], [], [], [], [], [], []
        adjmatrix_x = []
        
        # give credits only to the first piece - for WordPiece / BytePair Encoding
        index = 0
        total_tokens = 0
               
        for w, e, p, em, t, at in zip(words, entities, postags, entity_mention, triggers, argument_tags): 
            
            try:   
                tokens = self.tokenizer.tokenize(w) if w not in [self.CLS, self.SEP] else [w]
            except Exception as ex:
                print(ex)
                continue
                
            try:
                tokens_xx = self.tokenizer.convert_tokens_to_ids(tokens)
            except Exception as ex:
                print(ex)
                continue
                
                
            if w in [self.CLS, self.SEP]:
                is_head = [0]    
            else:
                is_head = [1] + [0] * (len(tokens) - 1)           

            
            p = [p] + [self.PAD] * (len(tokens) - 1)
            p = [postag2idx[postag] for postag in p]
            e = [e] + [[self.PAD]] * (len(tokens) - 1)
            e = [[entity2idx[entity] for entity in entities] for entities in e]
            em = [em] + [self.PAD] * (len(tokens) - 1)
            em = [entity2idx[e] for e in em]    #ners2id            
            t = [t] + [self.PAD] * (len(tokens) - 1)
            t = [trigger_tag2idx[trigger] for trigger in t]
            at = [at] + [self.PAD] * (len(tokens) - 1)
            at = [argument_tag2idx[arg] for arg in at]
 
            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head), \
            entity_mention_x.extend(em), triggers_y.extend(t), argument_tags_y.extend(at)
                    
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, entity_mention_x, triggers_y, argument_tags_y, arguments, seqlen, \
            head_indexes, words, adjmatrix, event_polarity, event_modality, deptags, gov_words


def pad(batch):
    """ A function to pad the input. """
    #pad_index = tokenizer.convert_tokens_to_ids(PAD)   ### to-do: add in for ROBERTA
    pad_index = 0

    tokens_x_2d, entities_x_3d, postags_x_2d, entity_mention_x_2d, triggers_y_2d, argument_tags_2d, arguments_2d, seqlens_1d, \
    head_indexes_2d, words_2d, adjmatrix, event_polarity, event_modality, deptags, gov_words = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()
    
    attention_mask = copy.deepcopy(tokens_x_2d)

    for i in range(len(tokens_x_2d)):
        attention_mask[i] = [1] * len(tokens_x_2d[i]) + [0] * (maxlen - len(tokens_x_2d[i])) 
        tokens_x_2d[i] = tokens_x_2d[i] + [pad_index] * (maxlen - len(tokens_x_2d[i]))
        postags_x_2d[i] = postags_x_2d[i] + [postag2idx[PAD]] * (maxlen - len(postags_x_2d[i]))
        entity_mention_x_2d[i] = entity_mention_x_2d[i] + [entity2idx[PAD]] * (maxlen - len(entity_mention_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger_tag2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        argument_tags_2d[i] = argument_tags_2d[i] + [argument_tag2idx[PAD]] * (maxlen - len(argument_tags_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]       
               

    return tokens_x_2d, entities_x_3d, postags_x_2d, entity_mention_x_2d, \
           triggers_y_2d, argument_tags_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, adjmatrix, event_polarity, event_modality, deptags, gov_words, attention_mask
           
           
def generateAdjMatrix(edgeJsonList, sent_len):
    """ Function to generate Adjacency matrix from Depedency Parse Tree """
    sparseAdjMatrixPos = [[], [], []]
    sparseAdjMatrixValues = []
    
    deptags = []
    gov_words = [0] * sent_len

    adjMatrix = [0] * sent_len
    for i in range(sent_len):
        adjMatrix[i] = [0] * sent_len
        
  
    def build_deptags(from_, to_, type_):
        dep_type = type_.split(':')
        deptags.append((from_, to_, deptags2idx[dep_type[0]]))

    def addedge(type_, from_, to_, value_):
        sparseAdjMatrixPos[0].append(type_)
        sparseAdjMatrixPos[1].append(from_)
        sparseAdjMatrixPos[2].append(to_)
        sparseAdjMatrixValues.append(value_)

    for edgeJson in edgeJsonList:
        ss = edgeJson.split("/")
        gov_idx = int(ss[-1].split("=")[-1])
        dep_idx = int(ss[-2].split("=")[-1])
        etype = ss[0]
        if etype == "root" or gov_idx == -1 or dep_idx == -1:
            continue
        addedge(0, gov_idx, dep_idx, 1.0)
        addedge(1, dep_idx, gov_idx, 1.0)
        build_deptags(gov_idx, dep_idx, etype)
        gov_words[dep_idx] = gov_idx + 1

    for i in range(sent_len):
        addedge(2, i, i, 1.0)   ###similar to padding, this is mapping to and from to the same index
        
    for idx, head_idx in enumerate(gov_words):
        real_head_idx = head_idx - 1
        adjMatrix[idx][real_head_idx] = 1
        adjMatrix[real_head_idx][idx] = 1
        adjMatrix[idx][idx] = 1

    return adjMatrix, deptags, gov_words

def combine_named_nominal_entities(ner, nominal):
    """ A function to named and nominal entities """
    combined_entity = []
    if len(ner) == len(nominal):        
        for idx, word in enumerate(nominal):
            if word == 'O' and ner[idx] != 'O':
                combined_entity.append(ner[idx])
            else:
                combined_entity.append(word)

    return combined_entity