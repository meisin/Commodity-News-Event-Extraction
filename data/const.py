
transformer_type = "bert"  ### to-do: need to parameterize transformer_type

from transformers import AutoTokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertConfig, RobertaConfig

OWN_TRANSFORMER_DIR = 'ComBERT/'

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, AutoTokenizer, 'bert-base-cased'),
    "combert": (BertConfig, BertModel, BertTokenizer, OWN_TRANSFORMER_DIR),
    "roberta": (RobertaConfig, RobertaModel, AutoTokenizer, 'roberta-base')
}

NONE = 'O'

PAD_BERT = "[PAD]"    #PAD: 0
UNK_BERT = "[UNK]"    #UNK: 100
CLS_BERT = '[CLS]'    #CLS : 101
SEP_BERT = '[SEP]'    #SEP: 102

PAD_ROBERTA = "<pad>"   #<pad>: 1
UNK_ROBERTA = "<unk>"   #<unk>: 3
CLS_ROBERTA = "<s>"     #<s> : 0
SEP_ROBERTA = "</s>"    #</s>: 2

INFINITY_NUMBER = 1e12

# event types
TRIGGERS = ['MOVEMENT-UP-GAIN', 'CAUSE-MOVEMENT-DOWN-LOSS', 'MOVEMENT-DOWN-LOSS', 'CIVIL-UNREST', 'EMBARGO', 'SLOW-WEAK',
            'CAUSE-MOVEMENT-UP-GAIN','GROW-STRONG', 'GEOPOLITICAL-TENSION','OVERSUPPLY','SHORTAGE', 'PROHIBITING', 
            'MOVEMENT-FLAT', 'SITUATION_DETERIORATE', 'TRADE-FINANCIAL-TENSION', 'POSITION-HIGH', 'POSITION-LOW',
            'NEGATIVE_SENTIMENT', 'CRISIS']

ARGUMENTS = ['NONE', 'ATTRIBUTE', 'ITEM', 'FINAL_VALUE', 'INITIAL_REFERENCE_POINT', 'PLACE', 'REFERENCE_POINT_TIME', 
             'DIFFERENCE', 'SUPPLIER_CONSUMER', 'IMPOSER', 'CONTRACT_DATE', 'TYPE', 'IMPOSEE',
             'IMPACTED_COUNTRIES', 'INITIAL_VALUE', 'DURATION', 'ACTIVITY', 'SITUATION', 'PARTICIPATING_COUNTRIES',
             'FORECASTER', 'FORECAST']

# entities
NOMINAL_ENTITIES = ['FINANCIAL_ATTRIBUTE','COMMODITY', 'DATE', 'COUNTRY', 'MONEY','ORGANIZATION','PRODUCTION_UNIT', 
            'PRICE_UNIT', 'GROUP', 'QUANTITY', 'NUMBER', 'DURATION', 'PERSON', 'OTHER_ACTIVITIES',
            'ECONOMIC_ITEM','LOCATION','PERCENTAGE', 'NATIONALITY','STATE_OR_PROVINCE', 'PHENOMENON', 
            'FORECAST_TARGET']

NER = ['DATE', 'LOCATION', 'MONEY', 'ORGANIZATION', 'PERCENT', 'PERSON', 'TIME', 'URL', 'COUNTRY', 'MISC', 'TITLE', 'DURATION',
       'NUMBER', 'NATIONALITY', 'CAUSE_OF_DEATH', 'STATE_OR_PROVINCE', 'ORDINAL', 'CITY', 'SET', 'RELIGION', 'IDEOLOGY'] 

ENTITIES = NOMINAL_ENTITIES + NER
ENTITIES = list(set(ENTITIES))  ### to remove duplicates

# 45 pos tags

POSTAGS = ['IN','NN','NNP','NNS','NNPS', 'NNP','DT','JJ','RB',',','.','CD','CD','EX', 'FW', 'VBG','VB','VBD','VBZ','VBN','VBP',
           'WDT','$','PRP','CC','TO','JJS','LS', 'MD', 'MD',"''",'JJR','PDT', 'POS','``',':','RP','-LRB-','-RRB-','PRP$','RB', 
           'RBR','RBS', 'SYM', 'TO', 'WRB','WP', 'WP$', 'UH', '(', ')']


VERB_FAMILY = ['VBG','VB','VBD','VBZ','VBN','VBP']

DEPTAGS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 
           'csubjpass', 'dep', 'det', 'discourse', 'dislocated', 'dobj', 'expl', 'foreign', 'goeswith', 'iobj', 'list', 'mark',
           'mwe', 'name', 'neg', 'nmod', 'nsubj', 'nsubjpass', 'nummod', 'parataxis', 'punct', 'remnant', 'reparandum', 
           'root', 'vocative', 'xcomp', 'ref', 'trigger_words', 'mod']
            ### trigger_words: 41

# Polarity & Modality
POLARITY = ['POSITIVE', 'NEGATIVE']
MODALITY = ['ASSERTED', 'OTHER']