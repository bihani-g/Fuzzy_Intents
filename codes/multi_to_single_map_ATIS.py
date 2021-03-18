#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from statistics import mean
import pandas as pd
from pandas import read_csv
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 


# In[2]:


single_data = pd.read_csv('./data/ATIS.csv')
single_data = single_data.drop_duplicates()
single_data.head()


# In[21]:


data_atis = pd.read_csv('./data/MixATIS.csv')
data_filtered = data_atis[['utterance', 'atis_abbreviation', 'atis_airfare', 'atis_airline', 'atis_ground_service', 'atis_flight']]
data_filt = data_filtered[(data_filtered.atis_abbreviation==1) | (data_filtered.atis_airfare==1) 	| (data_filtered.atis_airline==1)  | 
                          (data_filtered.atis_ground_service==1)  | (data_filtered.atis_flight==1)]
data_filt['sum'] = data_filt.iloc[:, 1:].sum(axis=1)
full_data = data_filt[data_filt['sum'] > 1].reset_index(drop = True)
full_data.head()


# In[15]:


full_data_int = full_data.drop(['utterance', 'sum'], axis=1)

actual_list = list(full_data_int.dot(full_data_int.columns+',').str[:-1])

actual_intents = pd.DataFrame(zip(full_data.utterance, actual_list), columns = ['utterance', 'actuals'])
new = actual_intents["actuals"].str.split(",", n = 2, expand = True) 
actual_intents['actual1'] = new[0]
actual_intents['actual2'] = new[1]
actual_intents['actual3'] = new[2]

actual_intents = actual_intents.drop(['actuals'], axis = 1)
actual_intents.head()


# In[19]:


unique_labels = list(set(single_data.intent_label))


# In[5]:


single_utt_db = {}

for intent in unique_labels:
    single_utt_db[intent] = single_data[single_data.intent_label == intent].utterance


# In[6]:


##Defining similarity functions

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_cosine_sim(str1, str2): 
  corpus = [str1,str2] 
  vectorizer = TfidfVectorizer()
  trsfm=vectorizer.fit_transform(corpus)
  # pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 0','Document 1'])
  
  return cosine_similarity(trsfm[0:1], trsfm)[0][1]


# In[115]:


##finding top 3 matching intents, sentences and similarity measures
def multi_best_intents_token_set(utt):
    sim_dict = {}
    sent_dict = {}

    for intent in unique_labels:
        sim_list = []
        sent_list = []
        for i in single_utt_db[intent]:
            sim_list.append(fuzz.token_set_ratio(i, utt))
            sent_list.append(i)
        sim_dict[intent] = sim_list
        sent_dict[intent] = sent_list
        

    max_sim = {}
    max_index = {}

    for intent in unique_labels:
        max_sim[intent] = max(sim_dict[intent])
        max_index[intent] = [i for i, j in enumerate(sim_dict[intent]) if j == max_sim[intent]]
    
    best_1 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[:1]]
    best_intent_1 = best_1[0][0]
    best_sim_1 = best_1[0][1]
    best_singles_1 = [sent_dict[best_intent_1][i] for i in max_index[best_intent_1]]

    best_2 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[1:2]]
    best_intent_2 = best_2[0][0]
    best_sim_2 = best_2[0][1]
    best_singles_2 = [sent_dict[best_intent_2][i] for i in max_index[best_intent_2]]

    best_3 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[2:3]]
    best_intent_3 = best_3[0][0]
    best_sim_3 = best_3[0][1]
    best_singles_3 = [sent_dict[best_intent_3][i] for i in max_index[best_intent_3]]
    
    return best_intent_1, best_intent_2, best_intent_3, best_sim_1, best_sim_2, best_sim_3, best_singles_1, best_singles_2, best_singles_3


# In[116]:


##finding top 3 matching intents, sentences and similarity measures
def multi_best_intents_partial(utt):
    sim_dict = {}
    sent_dict = {}

    for intent in unique_labels:
        sim_list = []
        sent_list = []
        for i in single_utt_db[intent]:
            sim_list.append(fuzz.partial_ratio(i, utt))
            sent_list.append(i)
        sim_dict[intent] = sim_list
        sent_dict[intent] = sent_list
        

    max_sim = {}
    max_index = {}

    for intent in unique_labels:
        max_sim[intent] = max(sim_dict[intent])
        max_index[intent] = [i for i, j in enumerate(sim_dict[intent]) if j == max_sim[intent]]
    
    best_1 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[:1]]
    best_intent_1 = best_1[0][0]
    best_sim_1 = best_1[0][1]
    best_singles_1 = [sent_dict[best_intent_1][i] for i in max_index[best_intent_1]]

    best_2 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[1:2]]
    best_intent_2 = best_2[0][0]
    best_sim_2 = best_2[0][1]
    best_singles_2 = [sent_dict[best_intent_2][i] for i in max_index[best_intent_2]]

    best_3 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[2:3]]
    best_intent_3 = best_3[0][0]
    best_sim_3 = best_3[0][1]
    best_singles_3 = [sent_dict[best_intent_3][i] for i in max_index[best_intent_3]]
    
    return best_intent_1, best_intent_2, best_intent_3, best_sim_1, best_sim_2, best_sim_3, best_singles_1, best_singles_2, best_singles_3


# In[117]:


##finding top 3 matching intents, sentences and similarity measures
def multi_best_intents_jaccard(utt):
    sim_dict = {}
    sent_dict = {}

    for intent in unique_labels:
        sim_list = []
        sent_list = []
        for i in single_utt_db[intent]:
            sim_list.append(get_jaccard_sim(i, utt))
            sent_list.append(i)
        sim_dict[intent] = sim_list
        sent_dict[intent] = sent_list
        

    max_sim = {}
    max_index = {}

    for intent in unique_labels:
        max_sim[intent] = max(sim_dict[intent])
        max_index[intent] = [i for i, j in enumerate(sim_dict[intent]) if j == max_sim[intent]]
    
    best_1 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[:1]]
    best_intent_1 = best_1[0][0]
    best_sim_1 = best_1[0][1]
    best_singles_1 = [sent_dict[best_intent_1][i] for i in max_index[best_intent_1]]

    best_2 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[1:2]]
    best_intent_2 = best_2[0][0]
    best_sim_2 = best_2[0][1]
    best_singles_2 = [sent_dict[best_intent_2][i] for i in max_index[best_intent_2]]

    best_3 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[2:3]]
    best_intent_3 = best_3[0][0]
    best_sim_3 = best_3[0][1]
    best_singles_3 = [sent_dict[best_intent_3][i] for i in max_index[best_intent_3]]
    
    return best_intent_1, best_intent_2, best_intent_3, best_sim_1, best_sim_2, best_sim_3, best_singles_1, best_singles_2, best_singles_3


# In[118]:


##finding top 3 matching intents, sentences and similarity measures
def multi_best_intents_cosine(utt):
    sim_dict = {}
    sent_dict = {}

    for intent in unique_labels:
        sim_list = []
        sent_list = []
        for i in single_utt_db[intent]:
            sim_list.append(get_cosine_sim(i, utt))
            sent_list.append(i)
        sim_dict[intent] = sim_list
        sent_dict[intent] = sent_list
        

    max_sim = {}
    max_index = {}

    for intent in unique_labels:
        max_sim[intent] = max(sim_dict[intent])
        max_index[intent] = [i for i, j in enumerate(sim_dict[intent]) if j == max_sim[intent]]
    
    best_1 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[:1]]
    best_intent_1 = best_1[0][0]
    best_sim_1 = best_1[0][1]
    best_singles_1 = [sent_dict[best_intent_1][i] for i in max_index[best_intent_1]]

    best_2 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[1:2]]
    best_intent_2 = best_2[0][0]
    best_sim_2 = best_2[0][1]
    best_singles_2 = [sent_dict[best_intent_2][i] for i in max_index[best_intent_2]]

    best_3 = [(k, v) for k, v in sorted(max_sim.items(), key=lambda item: item[1], reverse = True)[2:3]]
    best_intent_3 = best_3[0][0]
    best_sim_3 = best_3[0][1]
    best_singles_3 = [sent_dict[best_intent_3][i] for i in max_index[best_intent_3]]
    
    return best_intent_1, best_intent_2, best_intent_3, best_sim_1, best_sim_2, best_sim_3, best_singles_1, best_singles_2, best_singles_3


# In[120]:


##token set calculations
i1_list = []
i2_list = []
i3_list = []
sim1_list = []
sim2_list = []
sim3_list = []
sent1_list = []
sent2_list = []
sent3_list = []

i = 0
for utt in tqdm(full_data.utterance.iloc[0:1000]):
    i1, i2, i3, sim1, sim2, sim3, sent1, sent2, sent3 = multi_best_intents_token_set(utt)
    i1_list.append(i1)
    i2_list.append(i2)
    i3_list.append(i3)
    sim1_list.append(sim1)
    sim2_list.append(sim2)
    sim3_list.append(sim3)
    sent1_list.append(sent1)
    sent2_list.append(sent2)
    sent3_list.append(sent3)
    i = i+1

tokenset_ratio_map = pd.DataFrame(zip(full_data.utterance.iloc[0:1000], i1_list, i2_list, i3_list,
                                    sim1_list, sim2_list, sim3_list, sent1_list, sent2_list, sent3_list,
                                     actual_intents.actual1.iloc[0:1000], actual_intents.actual2.iloc[0:1000],
                                      actual_intents.actual3.iloc[0:1000]), 
                                 columns = ['utterance', 'intent_1', 'intent_2', 'intent_3', 
                                           'score_1', 'score_2', 'score_3', 'sent_1', 'sent_2', 'sent_3',
                                           'actual1', 'actual2', 'actual3'])

tokenset_ratio_map.to_csv('./results/tokenset_ratio_map_ATIS_1000.csv', index = False)
print("token set calculations done!")


# In[123]:


##partial ratio calculations
i1_list = []
i2_list = []
i3_list = []
sim1_list = []
sim2_list = []
sim3_list = []
sent1_list = []
sent2_list = []
sent3_list = []

i = 0
for utt in tqdm(full_data.utterance.iloc[0:1000]):
    i1, i2, i3, sim1, sim2, sim3, sent1, sent2, sent3 = multi_best_intents_partial(utt)
    i1_list.append(i1)
    i2_list.append(i2)
    i3_list.append(i3)
    sim1_list.append(sim1)
    sim2_list.append(sim2)
    sim3_list.append(sim3)
    sent1_list.append(sent1)
    sent2_list.append(sent2)
    sent3_list.append(sent3)
    i = i+1

partial_ratio_map = pd.DataFrame(zip(full_data.utterance.iloc[0:1000], i1_list, i2_list, i3_list,
                                    sim1_list, sim2_list, sim3_list, sent1_list, sent2_list, sent3_list,
                                    actual_intents.actual1.iloc[0:1000], actual_intents.actual2.iloc[0:1000],
                                      actual_intents.actual3.iloc[0:1000]), 
                                 columns = ['utterance', 'intent_1', 'intent_2', 'intent_3', 
                                           'score_1', 'score_2', 'score_3', 'sent_1', 'sent_2', 'sent_3',
                                           'actual1', 'actual2', 'actual3'])

partial_ratio_map.to_csv('./results/partial_ratio_map_ATIS_1000.csv', index = False)
print("partial ratio calculations done!")


# In[124]:


##jaccard sim calculations
i1_list = []
i2_list = []
i3_list = []
sim1_list = []
sim2_list = []
sim3_list = []
sent1_list = []
sent2_list = []
sent3_list = []

i = 0
for utt in tqdm(full_data.utterance.iloc[0:1000]):
    i1, i2, i3, sim1, sim2, sim3, sent1, sent2, sent3 = multi_best_intents_jaccard(utt)
    i1_list.append(i1)
    i2_list.append(i2)
    i3_list.append(i3)
    sim1_list.append(sim1)
    sim2_list.append(sim2)
    sim3_list.append(sim3)
    sent1_list.append(sent1)
    sent2_list.append(sent2)
    sent3_list.append(sent3)
    i = i+1
    print(i, "utterances processed...")

jaccard_ratio_map = pd.DataFrame(zip(full_data.utterance.iloc[0:1000], i1_list, i2_list, i3_list,
                                    sim1_list, sim2_list, sim3_list, sent1_list, sent2_list, sent3_list,
                                    actual_intents.actual1.iloc[0:1000], actual_intents.actual2.iloc[0:1000],
                                      actual_intents.actual3.iloc[0:1000]), 
                                 columns = ['utterance', 'intent_1', 'intent_2', 'intent_3', 
                                           'score_1', 'score_2', 'score_3', 'sent_1', 'sent_2', 'sent_3',
                                           'actual1', 'actual2', 'actual3'])

jaccard_ratio_map.to_csv('./results/jaccard_ratio_map_ATIS_1000.csv', index = False)
print("jaccard sim calculations done!")


# In[125]:


##cosine sim calculations
i1_list = []
i2_list = []
i3_list = []
sim1_list = []
sim2_list = []
sim3_list = []
sent1_list = []
sent2_list = []
sent3_list = []

i = 0
for utt in tqdm(full_data.utterance.iloc[0:1000]):
    i1, i2, i3, sim1, sim2, sim3, sent1, sent2, sent3 = multi_best_intents_cosine(utt)
    i1_list.append(i1)
    i2_list.append(i2)
    i3_list.append(i3)
    sim1_list.append(sim1)
    sim2_list.append(sim2)
    sim3_list.append(sim3)
    sent1_list.append(sent1)
    sent2_list.append(sent2)
    sent3_list.append(sent3)
    i = i+1

cosine_ratio_map = pd.DataFrame(zip(full_data.utterance.iloc[0:1000], i1_list, i2_list, i3_list,
                                    sim1_list, sim2_list, sim3_list, sent1_list, sent2_list, sent3_list,
                                   actual_intents.actual1.iloc[0:1000], actual_intents.actual2.iloc[0:1000],
                                      actual_intents.actual3.iloc[0:1000]), 
                                 columns = ['utterance', 'intent_1', 'intent_2', 'intent_3', 
                                           'score_1', 'score_2', 'score_3', 'sent_1', 'sent_2', 'sent_3',
                                           'actual1', 'actual2', 'actual3'])

cosine_ratio_map.to_csv('./results/cosine_ratio_map_ATIS_1000.csv', index = False)
print("cosine sim calculations done!")


# In[ ]:




