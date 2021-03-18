#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#softmax mem data
softmax_mem = pd.read_csv('./results/softmax_data/SI_softmax_ATIS.csv')
softmax_mem = softmax_mem.iloc[:,1:]
softmax_mem = softmax_mem.drop_duplicates()


# In[3]:


#hardmax mem data
hardmax_mem = pd.read_csv('./results/softmax_data/SI_hardmax_ATIS.csv')
hardmax_mem = hardmax_mem.iloc[:,1:]
hardmax_mem = hardmax_mem.drop_duplicates()


# In[4]:


token_df0 = pd.read_csv('./results/sim_data/tokenset_ratio_map_ATIS_0.csv')
partial_df0 = pd.read_csv('./results/sim_data/partial_ratio_map_ATIS_0.csv')
jaccard_df0 = pd.read_csv('./results/sim_data/jaccard_ratio_map_ATIS_0.csv')
cosine_df0 = pd.read_csv('./results/sim_data/cosine_ratio_map_ATIS_0.csv')


# In[5]:


token_df1 = pd.read_csv('./results/sim_data/tokenset_ratio_map_ATIS_1.csv')
partial_df1 = pd.read_csv('./results/sim_data/partial_ratio_map_ATIS_1.csv')
jaccard_df1 = pd.read_csv('./results/sim_data/jaccard_ratio_map_ATIS_1.csv')
cosine_df1 = pd.read_csv('./results/sim_data/cosine_ratio_map_ATIS_1.csv')


# In[6]:


token_df2 = pd.read_csv('./results/sim_data/tokenset_ratio_map_ATIS_2.csv')
partial_df2 = pd.read_csv('./results/sim_data/partial_ratio_map_ATIS_2.csv')
jaccard_df2 = pd.read_csv('./results/sim_data/jaccard_ratio_map_ATIS_2.csv')
cosine_df2 = pd.read_csv('./results/sim_data/cosine_ratio_map_ATIS_2.csv')


# In[7]:


token_df3 = pd.read_csv('./results/sim_data/tokenset_ratio_map_ATIS_3.csv')
partial_df3 = pd.read_csv('./results/sim_data/partial_ratio_map_ATIS_3.csv')
jaccard_df3 = pd.read_csv('./results/sim_data/jaccard_ratio_map_ATIS_3.csv')
cosine_df3 = pd.read_csv('./results/sim_data/cosine_ratio_map_ATIS_3.csv')


# In[8]:


token_df4 = pd.read_csv('./results/sim_data/tokenset_ratio_map_ATIS_4.csv')
partial_df4 = pd.read_csv('./results/sim_data/partial_ratio_map_ATIS_4.csv')
jaccard_df4 = pd.read_csv('./results/sim_data/jaccard_ratio_map_ATIS_4.csv')
cosine_df4 = pd.read_csv('./results/sim_data/cosine_ratio_map_ATIS_4.csv')


# In[9]:


def sim_setup(sim_dataset):
    sim_data = sim_dataset
    
    sim_data['sent_1'] = sim_data['sent_1'].str.strip("'[]''""")
    sim_data['sent_1'] = sim_data['sent_1'].str.strip(" ")
    sim_data['sent_1'] = sim_data['sent_1'].str.replace("'", "")
    sim_data['sent_1'] = sim_data['sent_1'].str.replace('"', '')


    sim_data['sent_2'] = sim_data['sent_2'].str.strip("'[]'""")
    sim_data['sent_2'] = sim_data['sent_2'].str.strip(" ")
    sim_data['sent_2'] = sim_data['sent_2'].str.replace("'", "")
    sim_data['sent_2'] = sim_data['sent_2'].str.replace('"', '')


    sim_data['sent_3'] = sim_data['sent_3'].str.strip("'[]'""")
    sim_data['sent_3'] = sim_data['sent_3'].str.strip(" ")
    sim_data['sent_3'] = sim_data['sent_3'].str.replace("'", "")
    sim_data['sent_3'] = sim_data['sent_3'].str.replace('"', '')
    
    return sim_data


# In[10]:


token_df0 = sim_setup(token_df0)
token_df1 = sim_setup(token_df1)
token_df2 = sim_setup(token_df2)
token_df3 = sim_setup(token_df3)
token_df4 = sim_setup(token_df4)


# In[11]:


partial_df0 = sim_setup(partial_df0)
partial_df1 = sim_setup(partial_df1)
partial_df2 = sim_setup(partial_df2)
partial_df3 = sim_setup(partial_df3)
partial_df4 = sim_setup(partial_df4)


# In[12]:


cosine_df0 = sim_setup(cosine_df0)
cosine_df1 = sim_setup(cosine_df1)
cosine_df2 = sim_setup(cosine_df2)
cosine_df3 = sim_setup(cosine_df3)
cosine_df4 = sim_setup(cosine_df4)


# In[13]:


jaccard_df0 = sim_setup(jaccard_df0)
jaccard_df1 = sim_setup(jaccard_df1)
jaccard_df2 = sim_setup(jaccard_df2)
jaccard_df3 = sim_setup(jaccard_df3)
jaccard_df4 = sim_setup(jaccard_df4)


# In[26]:


def member_map_mi(sim_data_type, softmax_data):
    
    sim_data = sim_data_type
    softmax_mem = softmax_data

    mix_list = []
    si_list = []
    intent_list = []
    score_list = []
    mem_list = []
    abb_l_list = []
    abb_m_list = []
    abb_h_list = []
    airfare_l_list = []
    airfare_m_list = []
    airfare_h_list = []
    airline_l_list = []
    airline_m_list = []
    airline_h_list = []
    flight_l_list = []
    flight_m_list = []
    flight_h_list = []
    gs_l_list = []
    gs_m_list = []
    gs_h_list = []


    for utt in sim_data.utterance:
        sent1 = []
        for i in sim_data[sim_data.utterance == utt].sent_1:
            sent1 = i.split(", ")
        sent2 = []
        for i in sim_data[sim_data.utterance == utt].sent_2:
            sent2 = i.split(", ")
        sent3 = []
        for i in sim_data[sim_data.utterance == utt].sent_3:
            sent3 = i.split(", ")
        
        intent1 = sim_data[sim_data.utterance == utt].intent_1.values[0]
        intent2 = sim_data[sim_data.utterance == utt].intent_2.values[0]
        intent3 = sim_data[sim_data.utterance == utt].intent_3.values[0]

        score1 = sim_data[sim_data.utterance == utt].score_1.values[0]
        score2 = sim_data[sim_data.utterance == utt].score_2.values[0]
        score3 = sim_data[sim_data.utterance == utt].score_3.values[0]
    
        for sent in sent1:
            for j in range(len(softmax_mem.utterance)):
                si = softmax_mem.utterance.iloc[j]
                if sent == si:
                    mix_list.append(utt)
                    si_list.append(sent)
                    intent_list.append(intent1)
                    score_list.append(score1)
                    abb_l_list.append(softmax_mem.atis_abb_low.iloc[j])
                    abb_m_list.append(softmax_mem.atis_abb_med.iloc[j])
                    abb_h_list.append(softmax_mem.atis_abb_high.iloc[j])
                    airfare_l_list.append(softmax_mem.atis_airfare_low.iloc[j])
                    airfare_m_list.append(softmax_mem.atis_airfare_med.iloc[j])
                    airfare_h_list.append(softmax_mem.atis_airfare_high.iloc[j])
                    airline_l_list.append(softmax_mem.atis_airline_low.iloc[j])
                    airline_m_list.append(softmax_mem.atis_airline_med.iloc[j])
                    airline_h_list.append(softmax_mem.atis_airline_high.iloc[j])
                    flight_l_list.append(softmax_mem.atis_flight_low.iloc[j])
                    flight_m_list.append(softmax_mem.atis_flight_med.iloc[j])
                    flight_h_list.append(softmax_mem.atis_flight_high.iloc[j])
                    gs_l_list.append(softmax_mem.atis_ground_service_low.iloc[j])
                    gs_m_list.append(softmax_mem.atis_ground_service_med.iloc[j])
                    gs_h_list.append(softmax_mem.atis_ground_service_high.iloc[j])

        for sent in sent2:
            for j in range(len(softmax_mem.utterance)):
                si = softmax_mem.utterance.iloc[j]
                if sent == si:
                    mix_list.append(utt)
                    si_list.append(sent)
                    intent_list.append(intent2)
                    score_list.append(score2)
                    abb_l_list.append(softmax_mem.atis_abb_low.iloc[j])
                    abb_m_list.append(softmax_mem.atis_abb_med.iloc[j])
                    abb_h_list.append(softmax_mem.atis_abb_high.iloc[j])
                    airfare_l_list.append(softmax_mem.atis_airfare_low.iloc[j])
                    airfare_m_list.append(softmax_mem.atis_airfare_med.iloc[j])
                    airfare_h_list.append(softmax_mem.atis_airfare_high.iloc[j])
                    airline_l_list.append(softmax_mem.atis_airline_low.iloc[j])
                    airline_m_list.append(softmax_mem.atis_airline_med.iloc[j])
                    airline_h_list.append(softmax_mem.atis_airline_high.iloc[j])
                    flight_l_list.append(softmax_mem.atis_flight_low.iloc[j])
                    flight_m_list.append(softmax_mem.atis_flight_med.iloc[j])
                    flight_h_list.append(softmax_mem.atis_flight_high.iloc[j])
                    gs_l_list.append(softmax_mem.atis_ground_service_low.iloc[j])
                    gs_m_list.append(softmax_mem.atis_ground_service_med.iloc[j])
                    gs_h_list.append(softmax_mem.atis_ground_service_high.iloc[j])

        for sent in sent3:
            for j in range(len(softmax_mem.utterance)):
                si = softmax_mem.utterance.iloc[j]
                if sent == si:
                    mix_list.append(utt)
                    si_list.append(sent)
                    intent_list.append(intent3)
                    score_list.append(score3)
                    abb_l_list.append(softmax_mem.atis_abb_low.iloc[j])
                    abb_m_list.append(softmax_mem.atis_abb_med.iloc[j])
                    abb_h_list.append(softmax_mem.atis_abb_high.iloc[j])
                    airfare_l_list.append(softmax_mem.atis_airfare_low.iloc[j])
                    airfare_m_list.append(softmax_mem.atis_airfare_med.iloc[j])
                    airfare_h_list.append(softmax_mem.atis_airfare_high.iloc[j])
                    airline_l_list.append(softmax_mem.atis_airline_low.iloc[j])
                    airline_m_list.append(softmax_mem.atis_airline_med.iloc[j])
                    airline_h_list.append(softmax_mem.atis_airline_high.iloc[j])
                    flight_l_list.append(softmax_mem.atis_flight_low.iloc[j])
                    flight_m_list.append(softmax_mem.atis_flight_med.iloc[j])
                    flight_h_list.append(softmax_mem.atis_flight_high.iloc[j])
                    gs_l_list.append(softmax_mem.atis_ground_service_low.iloc[j])
                    gs_m_list.append(softmax_mem.atis_ground_service_med.iloc[j])
                    gs_h_list.append(softmax_mem.atis_ground_service_high.iloc[j])
                    
        multi_mem_df = pd.DataFrame(zip(mix_list, si_list, intent_list, score_list, abb_l_list, abb_m_list, abb_h_list,
                               airfare_l_list, airfare_m_list, airfare_h_list, airline_l_list, airline_m_list, 
                               airline_h_list, flight_l_list, flight_m_list, flight_h_list, gs_l_list,
                               gs_m_list, gs_h_list), columns = ['multi', 'single', 'intent', 'sim_score', 
                                                                 'abb_l', 'abb_m', 'abb_h', 'airfare_l', 
                                                                'airfare_m', 'airfare_h', 'airline_l', 'airline_m',
                                                                'airline_h', 'flight_l', 'flight_m', 'flight_h',
                                                                'gs_l', 'gs_m', 'gs_h'])
        
    return multi_mem_df
    
    


# In[ ]:


##softmax 

##tokenset
token_soft_0 = member_map_mi(token_df0,softmax_mem)
token_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_token_0.csv', index = False)

token_soft_1 = member_map_mi(token_df1,softmax_mem)
token_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_token_1.csv', index = False)

token_soft_2 = member_map_mi(token_df2,softmax_mem)
token_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_token_2.csv', index = False)

token_soft_3 = member_map_mi(token_df3,softmax_mem)
token_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_token_3.csv', index = False)

token_soft_4 = member_map_mi(token_df4,softmax_mem)
token_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_token_4.csv', index = False)


##jaccard
jaccard_soft_0 = member_map_mi(jaccard_df0,softmax_mem)
jaccard_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_jaccard_0.csv', index = False)

jaccard_soft_1 = member_map_mi(jaccard_df1,softmax_mem)
jaccard_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_jaccard_1.csv', index = False)

jaccard_soft_2 = member_map_mi(jaccard_df2,softmax_mem)
jaccard_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_jaccard_2.csv', index = False)

jaccard_soft_3 = member_map_mi(jaccard_df3,softmax_mem)
jaccard_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_jaccard_3.csv', index = False)

jaccard_soft_4 = member_map_mi(jaccard_df4,softmax_mem)
jaccard_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_jaccard_4.csv', index = False)

##cosine
cosine_soft_0 = member_map_mi(cosine_df0,softmax_mem)
cosine_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_cosine_0.csv', index = False)

cosine_soft_1 = member_map_mi(cosine_df1,softmax_mem)
cosine_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_cosine_1.csv', index = False)

cosine_soft_2 = member_map_mi(cosine_df2,softmax_mem)
cosine_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_cosine_2.csv', index = False)

cosine_soft_3 = member_map_mi(cosine_df3,softmax_mem)
cosine_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_cosine_3.csv', index = False)

cosine_soft_4 = member_map_mi(cosine_df4,softmax_mem)
cosine_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_cosine_4.csv', index = False)

##partial
partial_soft_0 = member_map_mi(partial_df0,softmax_mem)
partial_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_partial_0.csv', index = False)

partial_soft_1 = member_map_mi(partial_df1,softmax_mem)
partial_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_partial_1.csv', index = False)

partial_soft_2 = member_map_mi(partial_df2,softmax_mem)
partial_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_partial_2.csv', index = False)

partial_soft_3 = member_map_mi(partial_df3,softmax_mem)
partial_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_partial_3.csv', index = False)

partial_soft_4 = member_map_mi(partial_df4,softmax_mem)
partial_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_soft_partial_4.csv', index = False)


# In[ ]:


##hardmax

##tokenset
token_hard_0 = member_map_mi(token_df0,hardmax_mem)
token_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_token_0.csv', index = False)

token_hard_1 = member_map_mi(token_df1,hardmax_mem)
token_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_token_1.csv', index = False)

token_hard_2 = member_map_mi(token_df2,hardmax_mem)
token_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_token_2.csv', index = False)

token_hard_3 = member_map_mi(token_df3,hardmax_mem)
token_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_token_3.csv', index = False)

token_hard_4 = member_map_mi(token_df4,hardmax_mem)
token_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_token_4.csv', index = False)


##jaccard
jaccard_hard_0 = member_map_mi(jaccard_df0,hardmax_mem)
jaccard_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_jaccard_0.csv', index = False)

jaccard_hard_1 = member_map_mi(jaccard_df1,hardmax_mem)
jaccard_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_jaccard_1.csv', index = False)

jaccard_hard_2 = member_map_mi(jaccard_df2,hardmax_mem)
jaccard_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_jaccard_2.csv', index = False)

jaccard_hard_3 = member_map_mi(jaccard_df3,hardmax_mem)
jaccard_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_jaccard_3.csv', index = False)

jaccard_hard_4 = member_map_mi(jaccard_df4,hardmax_mem)
jaccard_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_jaccard_4.csv', index = False)

##cosine
cosine_hard_0 = member_map_mi(cosine_df0,hardmax_mem)
cosine_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_cosine_0.csv', index = False)

cosine_hard_1 = member_map_mi(cosine_df1,hardmax_mem)
cosine_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_cosine_1.csv', index = False)

cosine_hard_2 = member_map_mi(cosine_df2,hardmax_mem)
cosine_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_cosine_2.csv', index = False)

cosine_hard_3 = member_map_mi(cosine_df3,hardmax_mem)
cosine_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_cosine_3.csv', index = False)

cosine_hard_4 = member_map_mi(cosine_df4,hardmax_mem)
cosine_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_cosine_4.csv', index = False)

##partial
partial_hard_0 = member_map_mi(partial_df0,hardmax_mem)
partial_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_partial_0.csv', index = False)

partial_hard_1 = member_map_mi(partial_df1,hardmax_mem)
partial_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_partial_1.csv', index = False)

partial_hard_2 = member_map_mi(partial_df2,hardmax_mem)
partial_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_partial_2.csv', index = False)

partial_hard_3 = member_map_mi(partial_df3,hardmax_mem)
partial_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_partial_3.csv', index = False)

partial_hard_4 = member_map_mi(partial_df4,hardmax_mem)
partial_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/ATIS_mem_hard_partial_4.csv', index = False)


# In[6]:


# #SI MI similarity
# # sim_data = pd.read_csv('./results/sim_data/'+sim_type+'_ratio_map_ATIS_'+str(k)+'.csv') 
# sim_data = pd.read_csv('./results/sim_data/partial_ratio_map_ATIS_1000.csv') 

# sim_data['sent_1'] = sim_data['sent_1'].str.strip("'[]''""")
# sim_data['sent_1'] = sim_data['sent_1'].str.strip(" ")
# sim_data['sent_1'] = sim_data['sent_1'].str.replace("'", "")
# sim_data['sent_1'] = sim_data['sent_1'].str.replace('"', '')


# sim_data['sent_2'] = sim_data['sent_2'].str.strip("'[]'""")
# sim_data['sent_2'] = sim_data['sent_2'].str.strip(" ")
# sim_data['sent_2'] = sim_data['sent_2'].str.replace("'", "")
# sim_data['sent_2'] = sim_data['sent_2'].str.replace('"', '')


# sim_data['sent_3'] = sim_data['sent_3'].str.strip("'[]'""")
# sim_data['sent_3'] = sim_data['sent_3'].str.strip(" ")
# sim_data['sent_3'] = sim_data['sent_3'].str.replace("'", "")
# sim_data['sent_3'] = sim_data['sent_3'].str.replace('"', '')


# In[3]:


# #map MI to mems

# mix_list = []
# si_list = []
# intent_list = []
# score_list = []
# mem_list = []
# abb_l_list = []
# abb_m_list = []
# abb_h_list = []
# airfare_l_list = []
# airfare_m_list = []
# airfare_h_list = []
# airline_l_list = []
# airline_m_list = []
# airline_h_list = []
# flight_l_list = []
# flight_m_list = []
# flight_h_list = []
# gs_l_list = []
# gs_m_list = []
# gs_h_list = []


# for utt in sim_data.utterance:
#     sent1 = []
#     for i in sim_data[sim_data.utterance == utt].sent_1:
#         sent1 = i.split(", ")
#     sent2 = []
#     for i in sim_data[sim_data.utterance == utt].sent_2:
#         sent2 = i.split(", ")
#     sent3 = []
#     for i in sim_data[sim_data.utterance == utt].sent_3:
#         sent3 = i.split(", ")
        
#     intent1 = sim_data[sim_data.utterance == utt].intent_1.values[0]
#     intent2 = sim_data[sim_data.utterance == utt].intent_2.values[0]
#     intent3 = sim_data[sim_data.utterance == utt].intent_3.values[0]

#     score1 = sim_data[sim_data.utterance == utt].score_1.values[0]
#     score2 = sim_data[sim_data.utterance == utt].score_2.values[0]
#     score3 = sim_data[sim_data.utterance == utt].score_3.values[0]
    
#     for sent in sent1:
#         for j in range(len(softmax_mem.utterance)):
#             si = softmax_mem.utterance.iloc[j]
#             if sent == si:
#                 mix_list.append(utt)
#                 si_list.append(sent)
#                 intent_list.append(intent1)
#                 score_list.append(score1)
#                 abb_l_list.append(softmax_mem.atis_abb_low.iloc[j])
#                 abb_m_list.append(softmax_mem.atis_abb_med.iloc[j])
#                 abb_h_list.append(softmax_mem.atis_abb_high.iloc[j])
#                 airfare_l_list.append(softmax_mem.atis_airfare_low.iloc[j])
#                 airfare_m_list.append(softmax_mem.atis_airfare_med.iloc[j])
#                 airfare_h_list.append(softmax_mem.atis_airfare_high.iloc[j])
#                 airline_l_list.append(softmax_mem.atis_airline_low.iloc[j])
#                 airline_m_list.append(softmax_mem.atis_airline_med.iloc[j])
#                 airline_h_list.append(softmax_mem.atis_airline_high.iloc[j])
#                 flight_l_list.append(softmax_mem.atis_flight_low.iloc[j])
#                 flight_m_list.append(softmax_mem.atis_flight_med.iloc[j])
#                 flight_h_list.append(softmax_mem.atis_flight_high.iloc[j])
#                 gs_l_list.append(softmax_mem.atis_ground_service_low.iloc[j])
#                 gs_m_list.append(softmax_mem.atis_ground_service_med.iloc[j])
#                 gs_h_list.append(softmax_mem.atis_ground_service_high.iloc[j])

#     for sent in sent2:
#         for j in range(len(softmax_mem.utterance)):
#             si = softmax_mem.utterance.iloc[j]
#             if sent == si:
#                 mix_list.append(utt)
#                 si_list.append(sent)
#                 intent_list.append(intent2)
#                 score_list.append(score2)
#                 abb_l_list.append(softmax_mem.atis_abb_low.iloc[j])
#                 abb_m_list.append(softmax_mem.atis_abb_med.iloc[j])
#                 abb_h_list.append(softmax_mem.atis_abb_high.iloc[j])
#                 airfare_l_list.append(softmax_mem.atis_airfare_low.iloc[j])
#                 airfare_m_list.append(softmax_mem.atis_airfare_med.iloc[j])
#                 airfare_h_list.append(softmax_mem.atis_airfare_high.iloc[j])
#                 airline_l_list.append(softmax_mem.atis_airline_low.iloc[j])
#                 airline_m_list.append(softmax_mem.atis_airline_med.iloc[j])
#                 airline_h_list.append(softmax_mem.atis_airline_high.iloc[j])
#                 flight_l_list.append(softmax_mem.atis_flight_low.iloc[j])
#                 flight_m_list.append(softmax_mem.atis_flight_med.iloc[j])
#                 flight_h_list.append(softmax_mem.atis_flight_high.iloc[j])
#                 gs_l_list.append(softmax_mem.atis_ground_service_low.iloc[j])
#                 gs_m_list.append(softmax_mem.atis_ground_service_med.iloc[j])
#                 gs_h_list.append(softmax_mem.atis_ground_service_high.iloc[j])

#     for sent in sent3:
#         for j in range(len(softmax_mem.utterance)):
#             si = softmax_mem.utterance.iloc[j]
#             if sent == si:
#                 mix_list.append(utt)
#                 si_list.append(sent)
#                 intent_list.append(intent3)
#                 score_list.append(score3)
#                 abb_l_list.append(softmax_mem.atis_abb_low.iloc[j])
#                 abb_m_list.append(softmax_mem.atis_abb_med.iloc[j])
#                 abb_h_list.append(softmax_mem.atis_abb_high.iloc[j])
#                 airfare_l_list.append(softmax_mem.atis_airfare_low.iloc[j])
#                 airfare_m_list.append(softmax_mem.atis_airfare_med.iloc[j])
#                 airfare_h_list.append(softmax_mem.atis_airfare_high.iloc[j])
#                 airline_l_list.append(softmax_mem.atis_airline_low.iloc[j])
#                 airline_m_list.append(softmax_mem.atis_airline_med.iloc[j])
#                 airline_h_list.append(softmax_mem.atis_airline_high.iloc[j])
#                 flight_l_list.append(softmax_mem.atis_flight_low.iloc[j])
#                 flight_m_list.append(softmax_mem.atis_flight_med.iloc[j])
#                 flight_h_list.append(softmax_mem.atis_flight_high.iloc[j])
#                 gs_l_list.append(softmax_mem.atis_ground_service_low.iloc[j])
#                 gs_m_list.append(softmax_mem.atis_ground_service_med.iloc[j])
#                 gs_h_list.append(softmax_mem.atis_ground_service_high.iloc[j])
                


# In[4]:


# multi_mem_df = pd.DataFrame(zip(mix_list, si_list, intent_list, score_list, abb_l_list, abb_m_list, abb_h_list,
#                                airfare_l_list, airfare_m_list, airfare_h_list, airline_l_list, airline_m_list, 
#                                airline_h_list, flight_l_list, flight_m_list, flight_h_list, gs_l_list,
#                                gs_m_list, gs_h_list), columns = ['multi', 'single', 'intent', 'sim_score', 
#                                                                  'abb_l', 'abb_m', 'abb_h', 'airfare_l', 
#                                                                 'airfare_m', 'airfare_h', 'airline_l', 'airline_m',
#                                                                 'airline_h', 'flight_l', 'flight_m', 'flight_h',
#                                                                 'gs_l', 'gs_m', 'gs_h'])


# In[5]:


# multi_mem_df.head()


# In[16]:


# # multi_mem_df.to_csv('./results/mix_mem_data/mix_mem_2/ATIS_multi_mem_'+sim_type+'_ratio_'+str(k)+'.csv', index = False)
# multi_mem_df.to_csv('./results/mix_mem_data/mix_mem_1/soft_ATIS_multi_mem_partial_ratio_1000.csv', index = False)


# In[ ]:




