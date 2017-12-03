
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.learner import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle
import pandas as pd
import numpy as np


# In[2]:


PATH='data/spooky-author-identification/'


get_ipython().run_line_magic('ls', '{PATH}')


# In[3]:


def save_data(df, file_train):
    trainData =""
    for idx, row in df.iterrows():
        data = row['text']
        if trainData == "":
            trainData= data
        else :
            trainData=trainData + " " + data

    file_train.write(trainData)
    file_train.close()
    return trainData


# In[4]:


file_train= open(f'{PATH}trainData.txt','w') 


# In[5]:


df_train = pd.read_csv(f'{PATH}train.csv')


# In[6]:


train_data= save_data(df_train,file_train)


# In[7]:


df_test = pd.read_csv(f'{PATH}test.csv')


# In[8]:


file_test= open(f'{PATH}testData.txt','w') 


# In[9]:


test_data= save_data(df_test,file_test)


# In[10]:


' '.join(spacy_tok(train_data))


# In[11]:


TEXT = data.Field(lower=True, tokenize=spacy_tok)


# In[12]:


TRN_PATH =  'trainData.txt'
VAL_PATH = 'testData.txt'
TRN = f'{PATH}trainData.txt'
VAL = f'{PATH}testData.txt'


# In[13]:


bs=2; bptt=70


# In[14]:


FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)


# In[15]:


md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)


# In[17]:


pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl','wb'))


# In[16]:


em_sz = 20  # size of each embedding vector
nh = 50     # number of hidden activations per layer
nl = 3       # number of layers


# In[17]:


opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


# In[18]:


learner = md.get_model(opt_fn, em_sz, nh, nl,
               dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3


# In[19]:


learner.fit(3e-3, 1, wds=1e-6, cycle_len=1, cycle_mult=1)


# In[20]:


learner.save_encoder('lang-model-spooky1')


# In[21]:


learner.load_encoder('lang-model-spooky1')


# In[22]:


TEXT = pickle.load(open(f'{PATH}models/TEXT.pkl','rb'))


# In[23]:


msk = np.random.rand(len(df_train)) < 0.9
df_val = df_train[~msk].reset_index()
df_train = df_train[msk].reset_index()


# In[24]:


df = {'train': df_train, 'val': df_val, 'test': df_test}


# In[25]:


LABEL = data.Field(sequential=False)


# In[26]:


df_train.head()


# In[28]:


df_test.head()


# In[29]:


df_val.head()


# In[57]:


for i in range(df_train.values[:,1].shape[0]):
    text = df_train.iloc[i].text
    label = df_train.iloc[i].author
    print(text)
    print(label)


# In[27]:


class PredictAuthorDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, label_field, dfs, **kwargs):
        fields = [("text", text_field), ("label", label_field)]
        examples = []
        for i in range(dfs[path].values[:,1].shape[0]):
            text = df_train.iloc[i].text
            label = df_train.iloc[i].author
            examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex): return len(ex.text)
    
    @classmethod
    def splits(cls, text_field, label_field, path,
               train, val, test, dfs, **kwargs):
        return super().splits(path,
            text_field=text_field, label_field=label_field,
            train=train, validation=val, test=test, dfs=dfs, **kwargs)


# In[28]:


splits = PredictAuthorDataset.splits(TEXT, LABEL, '',
                             train='train',
                             val='val', test='test', dfs=df)


# In[40]:


t = splits[1].examples[0]


# In[41]:


t.label, ' '.join(t.text[:10])


# In[44]:


t = splits[1].examples[2]


# In[45]:


t.label, ' '.join(t.text[:10])


# In[29]:


md2 = TextData.from_splits(PATH, splits, 1)


# In[47]:


df_train[:20]


# In[30]:


dropouti, dropout, wdrop, dropoute, dropouth = np.array([0.05,0.05,0.1,0.02,0.05])*6
m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl, 
           dropouti=dropouti, dropout=dropout, wdrop=wdrop,
                       dropoute=dropoute, dropouth=dropouth)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
m3.load_encoder(f'lang-model-spooky')
#m3.load_encoder(f'/home/ubuntu/data/spooky-author-identification/weights/imdb_adam3_c1_cl10_cyc_0')


# In[31]:


m3.clip=25.
lrs=np.array([1e-4,1e-3,1e-2])


# In[32]:


m3.freeze_to(-1)
m3.fit(lrs/2, 1, metrics=[accuracy])


# In[33]:


log_preds, y = m3.predict_with_targs(is_test=True)


# In[34]:


log_preds[:10],y[:10]


# In[35]:


preds = m3.predict()


# In[48]:


res = np.argmax(log_preds, axis=1)


# In[49]:


res[:10]


# In[50]:


y[:10]


# In[51]:


pre_preds,y = m3.predict_with_targs(True)


# In[52]:


preds =to_np(F.softmax(torch.from_numpy(pre_preds[:,1:])))


# In[53]:


preds.shape, len(splits[2].examples)


# In[54]:


preds[:10]

