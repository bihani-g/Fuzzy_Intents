#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[5]:


data_atis = pd.read_csv('./data/MixATIS.csv')
data_filtered = data_atis[['utterance', 'atis_abbreviation', 'atis_airfare', 'atis_airline', 'atis_ground_service', 'atis_flight']]
data_filt = data_filtered[(data_filtered.atis_abbreviation==1) | (data_filtered.atis_airfare==1) 	| (data_filtered.atis_airline==1)  | 
                          (data_filtered.atis_ground_service==1)  | (data_filtered.atis_flight==1)]
data_filt['sum'] = data_filt.iloc[:, 1:].sum(axis=1)
full_data = data_filt[data_filt['sum'] > 1].reset_index(drop = True)
actual = full_data.drop_duplicates()


# In[8]:


# data = pd.read_csv('./results/mix_mem_data/mix_mem_1/ATIS_multi_mem_'+sim_type+'_ratio_'+str(k)+'.csv')
# data = pd.read_csv('./results/mix_mem_data/mix_mem_1/ATIS_multi_mem_jaccard_ratio.csv')
# data = pd.read_csv('./results/mix_mem_data/mix_mem_1/soft_ATIS_multi_mem_tokenset_ratio_1000.csv')

def data_mem_acc(dataset_name):
    data = pd.read_csv('./results/mix_mem_data/mix_mem_final/'+dataset_name)
    data_new = data.copy()

    ##multiply scores with similarity
    data_new['abb_l_n'] = data_new['abb_l']*data_new['sim_score']
    data_new['abb_m_n'] = data_new['abb_m']*data_new['sim_score']
    data_new['abb_h_n'] = data_new['abb_h']*data_new['sim_score']
    data_new['airfare_l_n'] = data_new['airfare_l']*data_new['sim_score']
    data_new['airfare_m_n'] = data_new['airfare_m']*data_new['sim_score']
    data_new['airfare_h_n'] = data_new['airfare_h']*data_new['sim_score']
    data_new['airline_l_n'] = data_new['airline_l']*data_new['sim_score']
    data_new['airline_m_n'] = data_new['airline_m']*data_new['sim_score']
    data_new['airline_h_n'] = data_new['airline_h']*data_new['sim_score']
    data_new['flight_l_n'] = data_new['flight_l']*data_new['sim_score']
    data_new['flight_m_n'] = data_new['flight_m']*data_new['sim_score']
    data_new['flight_h_n'] = data_new['flight_h']*data_new['sim_score']
    data_new['gs_l_n'] = data_new['gs_l']*data_new['sim_score']
    data_new['gs_m_n'] = data_new['gs_m']*data_new['sim_score']
    data_new['gs_h_n'] = data_new['gs_h']*data_new['sim_score']

    list_abb_l = []
    list_abb_m = []
    list_abb_h = []
    list_airfare_l = []
    list_airfare_m = []
    list_airfare_h = []
    list_airline_l = []
    list_airline_m = []
    list_airline_h = []
    list_flight_l = []
    list_flight_m = []
    list_flight_h = []
    list_gs_l = []
    list_gs_m = []
    list_gs_h = []
    utt_list = []


##get max mem value in a fuzzy intent for each mixed utterance
    for utt in list(set(data_new.multi)):
        utt_list.append(utt)
        list_abb_l.append(max(data_new[data_new.multi == utt].abb_l))
        list_abb_m.append(max(data_new[data_new.multi == utt].abb_m))
        list_abb_h.append(max(data_new[data_new.multi == utt].abb_h))
        list_airfare_l.append(max(data_new[data_new.multi == utt].airfare_l))
        list_airfare_m.append(max(data_new[data_new.multi == utt].airfare_m))
        list_airfare_h.append(max(data_new[data_new.multi == utt].airfare_h))
        list_airline_l.append(max(data_new[data_new.multi == utt].airline_l))
        list_airline_m.append(max(data_new[data_new.multi == utt].airline_m))
        list_airline_h.append(max(data_new[data_new.multi == utt].airline_h))
        list_flight_l.append(max(data_new[data_new.multi == utt].flight_l))
        list_flight_m.append(max(data_new[data_new.multi == utt].flight_m))
        list_flight_h.append(max(data_new[data_new.multi == utt].flight_h))
        list_gs_l.append(max(data_new[data_new.multi == utt].gs_l))
        list_gs_m.append(max(data_new[data_new.multi == utt].gs_m))
        list_gs_h.append(max(data_new[data_new.multi == utt].gs_h))
    
    mem_df_2 = pd.DataFrame(zip(utt_list, list_abb_l, list_abb_m, list_abb_h,
                           list_airfare_l, list_airfare_m, list_airfare_h,
                           list_airline_l, list_airline_m, list_airline_h,
                           list_flight_l, list_flight_m, list_flight_h,
                           list_gs_l, list_gs_m, list_gs_h), columns = ['utt', 'abb_l', 'abb_m', 'abb_h', 'airfare_l', 
                                                                'airfare_m', 'airfare_h', 'airline_l', 'airline_m',
                                                                'airline_h', 'flight_l', 'flight_m', 'flight_h',
                                                                'gs_l', 'gs_m', 'gs_h'])

    ##for all intents, if prioritize high mem, then medium and then low
    abb_list = []
    utt_list = []
    for i in range(len(mem_df_2)):
    #abb
        utt_list.append(mem_df_2.utt.iloc[i])
        if (mem_df_2.abb_h.iloc[i] > 0.5):
            abb_list.append('high')
        elif ((mem_df_2.abb_h.iloc[i] <  0.5) & (mem_df_2.abb_m.iloc[i] > 0.5)):
            abb_list.append('medium')
        elif ((mem_df_2.abb_h.iloc[i] <  0.5) & (mem_df_2.abb_m.iloc[i] < 0.5) & (mem_df_2.abb_l.iloc[i] > 0.5)):
            abb_list.append('low')
    

    airfare_list = []
    for i in range(len(mem_df_2)):
    #abb
        if (mem_df_2.airfare_h.iloc[i] > 0.5):
            airfare_list.append('high')
        elif ((mem_df_2.airfare_h.iloc[i] <  0.5) & (mem_df_2.airfare_m.iloc[i] > 0.5)):
            airfare_list.append('medium')
        elif ((mem_df_2.airfare_h.iloc[i] <  0.5) & (mem_df_2.airfare_m.iloc[i] < 0.5) & (mem_df_2.airfare_l.iloc[i] > 0.5)):
            airfare_list.append('low')
        
    airline_list = []
    for i in range(len(mem_df_2)):
    #abb
        if (mem_df_2.airline_h.iloc[i] > 0.5):
            airline_list.append('high')
        elif ((mem_df_2.airline_h.iloc[i] <  0.5) & (mem_df_2.airline_m.iloc[i] > 0.5)):
            airline_list.append('medium')
        elif ((mem_df_2.airline_h.iloc[i] <  0.5) & (mem_df_2.airline_m.iloc[i] < 0.5) & (mem_df_2.airline_l.iloc[i] > 0.5)):
            airline_list.append('low')
    
    flight_list = []
    for i in range(len(mem_df_2)):
    #abb
        if (mem_df_2.flight_h.iloc[i] > 0.5):
            flight_list.append('high')
        elif ((mem_df_2.flight_h.iloc[i] <  0.5) & (mem_df_2.flight_m.iloc[i] > 0.5)):
            flight_list.append('medium')
        elif ((mem_df_2.flight_h.iloc[i] <  0.5) & (mem_df_2.flight_m.iloc[i] < 0.5) & (mem_df_2.flight_l.iloc[i] > 0.5)):
            flight_list.append('low')

    gs_list = []
    for i in range(len(mem_df_2)):
    #abb
        if (mem_df_2.gs_h.iloc[i] > 0.5):
            gs_list.append('high')
        elif ((mem_df_2.gs_h.iloc[i] <  0.5) & (mem_df_2.gs_m.iloc[i] > 0.5)):
            gs_list.append('medium')
        elif ((mem_df_2.gs_h.iloc[i] <  0.5) & (mem_df_2.gs_m.iloc[i] < 0.5) & (mem_df_2.gs_l.iloc[i] > 0.5)):
            gs_list.append('low')

    mem_df_3 = pd.DataFrame(zip(utt_list, abb_list, airfare_list, airline_list, flight_list, gs_list),
                       columns = ['utt', 'abb', 'airfare', 'airline', 'flight', 'gs'])

    ##compare with actual data
    hit_list = []
    for i in range(len(mem_df_3.utt)):
        hits = 0
        utt = mem_df_3.utt.iloc[i]
        abb_actual = actual[actual.utterance == utt].atis_abbreviation.values
        airfare_actual = actual[actual.utterance == utt].atis_airfare.values
        airline_actual = actual[actual.utterance == utt].atis_airline.values
        flight_actual = actual[actual.utterance == utt].atis_flight.values
        gs_actual = actual[actual.utterance == utt].atis_ground_service.values
    
        if ((abb_actual == 1) & (mem_df_3.abb.iloc[i] == 'high')):
            hits = hits + 1
    
        if ((abb_actual == 0) & (mem_df_3.abb.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((airfare_actual == 1) & (mem_df_3.airfare.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((airfare_actual == 0) & (mem_df_3.airfare.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((airline_actual == 1) & (mem_df_3.airline.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((airline_actual == 0) & (mem_df_3.airline.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((flight_actual == 1) & (mem_df_3.flight.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((flight_actual == 0) & (mem_df_3.flight.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((gs_actual == 1) & (mem_df_3.gs.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((gs_actual == 0) & (mem_df_3.gs.iloc[i] == 'low')):
            hits = hits + 1

        hit_list.append(hits)

    mem_df_3['hits'] = hit_list
    mem_df_3['hit_acc'] = mem_df_3['hits']/5

    mem_df_3['hit_acc01'] = [1 if x == 1 else 0 for x in mem_df_3['hit_acc']]

    from statistics import mean
    # print('For '+ str(k) + ' SI instances in database and '+ sim_type + ' similarity;')
    print(dataset_name)
    print('Avg Accuracy per utterance:', mean(mem_df_3['hit_acc']))
    
    return mem_df_3


# In[15]:


#hard
print('hard membership')
#cosine
print('cosine')

mem_hard_cosine_0 = data_mem_acc('ATIS_mem_hard_cosine_0.csv')
mem_hard_cosine_1 = data_mem_acc('ATIS_mem_hard_cosine_1.csv')
mem_hard_cosine_2 = data_mem_acc('ATIS_mem_hard_cosine_2.csv')
mem_hard_cosine_3 = data_mem_acc('ATIS_mem_hard_cosine_3.csv')
mem_hard_cosine_4 = data_mem_acc('ATIS_mem_hard_cosine_4.csv')

#jaccard
print('jaccard')

mem_hard_jaccard_0 = data_mem_acc('ATIS_mem_hard_jaccard_0.csv')
mem_hard_jaccard_1 = data_mem_acc('ATIS_mem_hard_jaccard_1.csv')
mem_hard_jaccard_2 = data_mem_acc('ATIS_mem_hard_jaccard_2.csv')
mem_hard_jaccard_3 = data_mem_acc('ATIS_mem_hard_jaccard_3.csv')
mem_hard_jaccard_4 = data_mem_acc('ATIS_mem_hard_jaccard_4.csv')

#partial
print('partial')

mem_hard_partial_0 = data_mem_acc('ATIS_mem_hard_partial_0.csv')
mem_hard_partial_1 = data_mem_acc('ATIS_mem_hard_partial_1.csv')
mem_hard_partial_2 = data_mem_acc('ATIS_mem_hard_partial_2.csv')
mem_hard_partial_3 = data_mem_acc('ATIS_mem_hard_partial_3.csv')
mem_hard_partial_4 = data_mem_acc('ATIS_mem_hard_partial_4.csv')

#token
print('token')

mem_hard_token_0 = data_mem_acc('ATIS_mem_hard_token_0.csv')
mem_hard_token_1 = data_mem_acc('ATIS_mem_hard_token_1.csv')
mem_hard_token_2 = data_mem_acc('ATIS_mem_hard_token_2.csv')
mem_hard_token_3 = data_mem_acc('ATIS_mem_hard_token_3.csv')
mem_hard_token_4 = data_mem_acc('ATIS_mem_hard_token_4.csv')


# In[ ]:


#soft
print('soft membership')
#cosine
print('cosine')

mem_soft_cosine_0 = data_mem_acc('ATIS_mem_soft_cosine_0.csv')
mem_soft_cosine_1 = data_mem_acc('ATIS_mem_soft_cosine_1.csv')
mem_soft_cosine_2 = data_mem_acc('ATIS_mem_soft_cosine_2.csv')
mem_soft_cosine_3 = data_mem_acc('ATIS_mem_soft_cosine_3.csv')
mem_soft_cosine_4 = data_mem_acc('ATIS_mem_soft_cosine_4.csv')

#jaccard
print('jaccard')

mem_soft_jaccard_0 = data_mem_acc('ATIS_mem_soft_jaccard_0.csv')
mem_soft_jaccard_1 = data_mem_acc('ATIS_mem_soft_jaccard_1.csv')
mem_soft_jaccard_2 = data_mem_acc('ATIS_mem_soft_jaccard_2.csv')
mem_soft_jaccard_3 = data_mem_acc('ATIS_mem_soft_jaccard_3.csv')
mem_soft_jaccard_4 = data_mem_acc('ATIS_mem_soft_jaccard_4.csv')

#partial
print('partial')

mem_soft_partial_0 = data_mem_acc('ATIS_mem_soft_partial_0.csv')
mem_soft_partial_1 = data_mem_acc('ATIS_mem_soft_partial_1.csv')
mem_soft_partial_2 = data_mem_acc('ATIS_mem_soft_partial_2.csv')
mem_soft_partial_3 = data_mem_acc('ATIS_mem_soft_partial_3.csv')
mem_soft_partial_4 = data_mem_acc('ATIS_mem_soft_partial_4.csv')

#token
print('token')

mem_soft_token_0 = data_mem_acc('ATIS_mem_soft_token_0.csv')
mem_soft_token_1 = data_mem_acc('ATIS_mem_soft_token_1.csv')
mem_soft_token_2 = data_mem_acc('ATIS_mem_soft_token_2.csv')
mem_soft_token_3 = data_mem_acc('ATIS_mem_soft_token_3.csv')
mem_soft_token_4 = data_mem_acc('ATIS_mem_soft_token_4.csv')


# In[14]:


import os

os.chdir('/media/disk4/fuzzy_intents/results/mix_mem_data/mix_mem_final/final_mem/')


# In[ ]:


#converting membership data to csv

#hard
#cosine
mem_hard_cosine_0.to_csv('ATIS_mem_hard_cosine_0_r.csv')
mem_hard_cosine_1.to_csv('ATIS_mem_hard_cosine_1_r.csv')
mem_hard_cosine_2.to_csv('ATIS_mem_hard_cosine_2_r.csv')
mem_hard_cosine_3.to_csv('ATIS_mem_hard_cosine_3_r.csv')
mem_hard_cosine_4.to_csv('ATIS_mem_hard_cosine_4_r.csv')

#jaccard
mem_hard_jaccard_0.to_csv('ATIS_mem_hard_jaccard_0_r.csv')
mem_hard_jaccard_1.to_csv('ATIS_mem_hard_jaccard_1_r.csv')
mem_hard_jaccard_2.to_csv('ATIS_mem_hard_jaccard_2_r.csv')
mem_hard_jaccard_3.to_csv('ATIS_mem_hard_jaccard_3_r.csv')
mem_hard_jaccard_4.to_csv('ATIS_mem_hard_jaccard_4_r.csv')

#partial
mem_hard_partial_0.to_csv('ATIS_mem_hard_partial_0_r.csv')
mem_hard_partial_1.to_csv('ATIS_mem_hard_partial_1_r.csv')
mem_hard_partial_2.to_csv('ATIS_mem_hard_partial_2_r.csv')
mem_hard_partial_3.to_csv('ATIS_mem_hard_partial_3_r.csv')
mem_hard_partial_4.to_csv('ATIS_mem_hard_partial_4_r.csv')

#token
mem_hard_token_0.to_csv('ATIS_mem_hard_token_0_r.csv')
mem_hard_token_1.to_csv('ATIS_mem_hard_token_1_r.csv')
mem_hard_token_2.to_csv('ATIS_mem_hard_token_2_r.csv')
mem_hard_token_3.to_csv('ATIS_mem_hard_token_3_r.csv')
mem_hard_token_4.to_csv('ATIS_mem_hard_token_4_r.csv')

##################################################################

#soft
#cosine
mem_soft_cosine_0.to_csv('ATIS_mem_soft_cosine_0_r.csv')
mem_soft_cosine_1.to_csv('ATIS_mem_soft_cosine_1_r.csv')
mem_soft_cosine_2.to_csv('ATIS_mem_soft_cosine_2_r.csv')
mem_soft_cosine_3.to_csv('ATIS_mem_soft_cosine_3_r.csv')
mem_soft_cosine_4.to_csv('ATIS_mem_soft_cosine_4_r.csv')

#jaccard
mem_soft_jaccard_0.to_csv('ATIS_mem_soft_jaccard_0_r.csv')
mem_soft_jaccard_1.to_csv('ATIS_mem_soft_jaccard_1_r.csv')
mem_soft_jaccard_2.to_csv('ATIS_mem_soft_jaccard_2_r.csv')
mem_soft_jaccard_3.to_csv('ATIS_mem_soft_jaccard_3_r.csv')
mem_soft_jaccard_4.to_csv('ATIS_mem_soft_jaccard_4_r.csv')

#partial
mem_soft_partial_0.to_csv('ATIS_mem_soft_partial_0_r.csv')
mem_soft_partial_1.to_csv('ATIS_mem_soft_partial_1_r.csv')
mem_soft_partial_2.to_csv('ATIS_mem_soft_partial_2_r.csv')
mem_soft_partial_3.to_csv('ATIS_mem_soft_partial_3_r.csv')
mem_soft_partial_4.to_csv('ATIS_mem_soft_partial_4_r.csv')

#token
mem_soft_token_0.to_csv('ATIS_mem_soft_token_0_r.csv')
mem_soft_token_1.to_csv('ATIS_mem_soft_token_1_r.csv')
mem_soft_token_2.to_csv('ATIS_mem_soft_token_2_r.csv')
mem_soft_token_3.to_csv('ATIS_mem_soft_token_3_r.csv')
mem_soft_token_4.to_csv('ATIS_mem_soft_token_4_r.csv')


# In[ ]:


print('done!')

