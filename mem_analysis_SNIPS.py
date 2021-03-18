#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from statistics import mean


# In[ ]:


# import sys
# k = int(sys.argv[1])
# sim_type = str(sys.argv[2])


# In[3]:


data_atis = pd.read_csv('./data/MixSNIPS.csv')
data_filt = data_atis
data_filt['sum'] = data_filt.iloc[:, 1:].sum(axis=1)
full_data = data_filt[data_filt['sum'] > 1].reset_index(drop = True)
actual = full_data.drop_duplicates()


# In[24]:


# # data = pd.read_csv('./results/mix_mem_data/mix_mem_1/SNIPS_multi_mem_tokenset_ratio.csv')
# data = pd.read_csv('./results/mix_mem_data/mix_mem_1/soft_SNIPS_multi_mem_cosine_ratio_1000.csv')


# In[25]:


# data = pd.read_csv('./results/mix_mem_data/mix_mem_1/ATIS_multi_mem_'+sim_type+'_ratio_'+str(k)+'.csv')
# data = pd.read_csv('./results/mix_mem_data/mix_mem_1/SNIPS_multi_mem_jaccard_ratio.csv')
def data_mem_acc(dataset_name):
    data = pd.read_csv('./results/mix_mem_data/mix_mem_final/'+dataset_name)
    data_new = data.copy()
    ##multiply scores with similarity
    data_new['addtoplay_l_n'] = data_new['addtoplay_l']*data_new['sim_score']
    data_new['addtoplay_m_n'] = data_new['addtoplay_m']*data_new['sim_score']
    data_new['addtoplay_h_n'] = data_new['addtoplay_h']*data_new['sim_score']
    data_new['bookrestro_l_n'] = data_new['bookrestro_l']*data_new['sim_score']
    data_new['bookrestro_m_n'] = data_new['bookrestro_m']*data_new['sim_score']
    data_new['bookrestro_h_n'] = data_new['bookrestro_h']*data_new['sim_score']
    data_new['getweath_l_n'] = data_new['getweath_l']*data_new['sim_score']
    data_new['getweath_m_n'] = data_new['getweath_m']*data_new['sim_score']
    data_new['getweath_h_n'] = data_new['getweath_h']*data_new['sim_score']
    data_new['playmusic_l_n'] = data_new['playmusic_l']*data_new['sim_score']
    data_new['playmusic_m_n'] = data_new['playmusic_m']*data_new['sim_score']
    data_new['playmusic_h_n'] = data_new['playmusic_h']*data_new['sim_score']
    data_new['ratebook_l_n'] = data_new['ratebook_l']*data_new['sim_score']
    data_new['ratebook_m_n'] = data_new['ratebook_m']*data_new['sim_score']
    data_new['ratebook_h_n'] = data_new['ratebook_h']*data_new['sim_score']
    data_new['searchcreat_l_n'] = data_new['searchcreat_l']*data_new['sim_score']
    data_new['searchcreat_m_n'] = data_new['searchcreat_m']*data_new['sim_score']
    data_new['searchcreat_h_n'] = data_new['searchcreat_h']*data_new['sim_score']
    data_new['searchscreen_l_n'] = data_new['searchscreen_l']*data_new['sim_score']
    data_new['searchscreen_m_n'] = data_new['searchscreen_m']*data_new['sim_score']
    data_new['searchscreen_h_n'] = data_new['searchscreen_h']*data_new['sim_score']

    list_addtoplay_l = []
    list_addtoplay_m = []
    list_addtoplay_h = []
    list_bookrestro_l = []
    list_bookrestro_m = []
    list_bookrestro_h = []
    list_getweath_l = []
    list_getweath_m = []
    list_getweath_h = []
    list_playmusic_l = []
    list_playmusic_m = []
    list_playmusic_h = []
    list_ratebook_l = []
    list_ratebook_m = []
    list_ratebook_h = []
    list_searchcreat_l = []
    list_searchcreat_m = []
    list_searchcreat_h = []
    list_searchscreen_l = []
    list_searchscreen_m = []
    list_searchscreen_h = []

    utt_list = []


    ##get max mem value in a fuzzy intent for each mixed utterance
    for utt in list(set(data_new.multi)):
        utt_list.append(utt)
        list_addtoplay_l.append(max(data_new[data_new.multi == utt].addtoplay_l))
        list_addtoplay_m.append(max(data_new[data_new.multi == utt].addtoplay_m))
        list_addtoplay_h.append(max(data_new[data_new.multi == utt].addtoplay_h))
        list_bookrestro_l.append(max(data_new[data_new.multi == utt].bookrestro_l))
        list_bookrestro_m.append(max(data_new[data_new.multi == utt].bookrestro_m))
        list_bookrestro_h.append(max(data_new[data_new.multi == utt].bookrestro_h))
        list_getweath_l.append(max(data_new[data_new.multi == utt].getweath_l))
        list_getweath_m.append(max(data_new[data_new.multi == utt].getweath_m))
        list_getweath_h.append(max(data_new[data_new.multi == utt].getweath_h))
        list_playmusic_l.append(max(data_new[data_new.multi == utt].playmusic_l))
        list_playmusic_m.append(max(data_new[data_new.multi == utt].playmusic_m))
        list_playmusic_h.append(max(data_new[data_new.multi == utt].playmusic_h))
        list_ratebook_l.append(max(data_new[data_new.multi == utt].ratebook_l))
        list_ratebook_m.append(max(data_new[data_new.multi == utt].ratebook_m))
        list_ratebook_h.append(max(data_new[data_new.multi == utt].ratebook_h))
        list_searchcreat_l.append(max(data_new[data_new.multi == utt].searchcreat_l))
        list_searchcreat_m.append(max(data_new[data_new.multi == utt].searchcreat_m))
        list_searchcreat_h.append(max(data_new[data_new.multi == utt].searchcreat_h))
        list_searchscreen_l.append(max(data_new[data_new.multi == utt].searchscreen_l))
        list_searchscreen_m.append(max(data_new[data_new.multi == utt].searchscreen_m))
        list_searchscreen_h.append(max(data_new[data_new.multi == utt].searchscreen_h))
    
    mem_df_2 = pd.DataFrame(zip(utt_list, list_addtoplay_l, list_addtoplay_m, list_addtoplay_h,
                           list_bookrestro_l, list_bookrestro_m, list_bookrestro_h,
                           list_getweath_l, list_getweath_m, list_getweath_h,
                           list_playmusic_l, list_playmusic_m, list_playmusic_h,
                           list_ratebook_l, list_ratebook_m, list_ratebook_h,
                           list_searchcreat_l, list_searchcreat_m, list_searchcreat_h,
                           list_searchscreen_l, list_searchscreen_m, list_searchscreen_h), columns = ['utt', 'addtoplay_l', 'addtoplay_m', 'addtoplay_h', 'bookrestro_l', 
                                                                'bookrestro_m', 'bookrestro_h', 'getweath_l', 'getweath_m',
                                                                'getweath_h', 'playmusic_l', 'playmusic_m', 'playmusic_h',
                                                                'ratebook_l', 'ratebook_m', 'ratebook_h',
                                                                'searchcreat_l', 'searchcreat_m', 'searchcreat_h',
                                                                'searchscreen_l', 'searchscreen_m', 'searchscreen_h'])

    ##for all intents, if prioritize high mem, then medium and then low
    addtoplay_list = []
    utt_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        utt_list.append(mem_df_2.utt.iloc[i])
        if (mem_df_2.addtoplay_h.iloc[i] > 0.5):
            addtoplay_list.append('high')
        elif ((mem_df_2.addtoplay_h.iloc[i] <  0.5) & (mem_df_2.addtoplay_m.iloc[i] > 0.5)):
            addtoplay_list.append('medium')
        elif ((mem_df_2.addtoplay_h.iloc[i] <  0.5) & (mem_df_2.addtoplay_m.iloc[i] < 0.5) & (mem_df_2.addtoplay_l.iloc[i] > 0.5)):
            addtoplay_list.append('low')
    

    bookrestro_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        if (mem_df_2.bookrestro_h.iloc[i] > 0.5):
            bookrestro_list.append('high')
        elif ((mem_df_2.bookrestro_h.iloc[i] <  0.5) & (mem_df_2.bookrestro_m.iloc[i] > 0.5)):
            bookrestro_list.append('medium')
        elif ((mem_df_2.bookrestro_h.iloc[i] <  0.5) & (mem_df_2.bookrestro_m.iloc[i] < 0.5) & (mem_df_2.bookrestro_l.iloc[i] > 0.5)):
            bookrestro_list.append('low')
        
    getweath_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        if (mem_df_2.getweath_h.iloc[i] > 0.5):
            getweath_list.append('high')
        elif ((mem_df_2.getweath_h.iloc[i] <  0.5) & (mem_df_2.getweath_m.iloc[i] > 0.5)):
            getweath_list.append('medium')
        elif ((mem_df_2.getweath_h.iloc[i] <  0.5) & (mem_df_2.getweath_m.iloc[i] < 0.5) & (mem_df_2.getweath_l.iloc[i] > 0.5)):
            getweath_list.append('low')
    
    playmusic_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        if (mem_df_2.playmusic_h.iloc[i] > 0.5):
            playmusic_list.append('high')
        elif ((mem_df_2.playmusic_h.iloc[i] <  0.5) & (mem_df_2.playmusic_m.iloc[i] > 0.5)):
            playmusic_list.append('medium')
        elif ((mem_df_2.playmusic_h.iloc[i] <  0.5) & (mem_df_2.playmusic_m.iloc[i] < 0.5) & (mem_df_2.playmusic_l.iloc[i] > 0.5)):
            playmusic_list.append('low')

    ratebook_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        if (mem_df_2.ratebook_h.iloc[i] > 0.5):
            ratebook_list.append('high')
        elif ((mem_df_2.ratebook_h.iloc[i] <  0.5) & (mem_df_2.ratebook_m.iloc[i] > 0.5)):
            ratebook_list.append('medium')
        elif ((mem_df_2.ratebook_h.iloc[i] <  0.5) & (mem_df_2.ratebook_m.iloc[i] < 0.5) & (mem_df_2.ratebook_l.iloc[i] > 0.5)):
            ratebook_list.append('low')
        
    searchcreat_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        if (mem_df_2.searchcreat_h.iloc[i] > 0.5):
            searchcreat_list.append('high')
        elif ((mem_df_2.searchcreat_h.iloc[i] <  0.5) & (mem_df_2.searchcreat_m.iloc[i] > 0.5)):
            searchcreat_list.append('medium')
        elif ((mem_df_2.searchcreat_h.iloc[i] <  0.5) & (mem_df_2.searchcreat_m.iloc[i] < 0.5) & (mem_df_2.searchcreat_l.iloc[i] > 0.5)):
            searchcreat_list.append('low')

        
    searchscreen_list = []
    for i in range(len(mem_df_2)):
        #addtoplay
        if (mem_df_2.searchscreen_h.iloc[i] > 0.5):
            searchscreen_list.append('high')
        elif ((mem_df_2.searchscreen_h.iloc[i] <  0.5) & (mem_df_2.searchscreen_m.iloc[i] > 0.5)):
            searchscreen_list.append('medium')
        elif ((mem_df_2.searchscreen_h.iloc[i] <  0.5) & (mem_df_2.searchscreen_m.iloc[i] < 0.5) & (mem_df_2.searchscreen_l.iloc[i] > 0.5)):
            searchscreen_list.append('low')
        
        
    mem_df_3 = pd.DataFrame(zip(utt_list, addtoplay_list, bookrestro_list, getweath_list, playmusic_list, ratebook_list,
                           searchcreat_list, searchscreen_list),
                       columns = ['utt', 'addtoplay', 'bookrestro', 'getweath', 'playmusic', 'ratebook', 'searchcreat',
                                 'searchscreen'])

    ##compare with actual data
    hit_list = []
    for i in range(len(mem_df_3.utt)):
        hits = 0
        utt = mem_df_3.utt.iloc[i]
        addtoplay_actual = actual[actual.utterance == utt].AddToPlaylist.values
        bookrestro_actual = actual[actual.utterance == utt].BookRestaurant.values
        getweath_actual = actual[actual.utterance == utt].GetWeather.values
        playmusic_actual = actual[actual.utterance == utt].PlayMusic.values
        ratebook_actual = actual[actual.utterance == utt].RateBook.values
        searchcreat_actual = actual[actual.utterance == utt].SearchCreativeWork.values
        searchscreen_actual = actual[actual.utterance == utt].SearchScreeningEvent.values
    
        if ((addtoplay_actual == 1) & (mem_df_3.addtoplay.iloc[i] == 'high')):
            hits = hits + 1
    
        if ((addtoplay_actual == 0) & (mem_df_3.addtoplay.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((bookrestro_actual == 1) & (mem_df_3.bookrestro.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((bookrestro_actual == 0) & (mem_df_3.bookrestro.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((getweath_actual == 1) & (mem_df_3.getweath.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((getweath_actual == 0) & (mem_df_3.getweath.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((playmusic_actual == 1) & (mem_df_3.playmusic.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((playmusic_actual == 0) & (mem_df_3.playmusic.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((ratebook_actual == 1) & (mem_df_3.ratebook.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((ratebook_actual == 0) & (mem_df_3.ratebook.iloc[i] == 'low')):
            hits = hits + 1

        if ((searchcreat_actual == 1) & (mem_df_3.searchcreat.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((searchcreat_actual == 0) & (mem_df_3.searchcreat.iloc[i] == 'low')):
            hits = hits + 1
    
        if ((searchscreen_actual == 1) & (mem_df_3.searchscreen.iloc[i] == 'high')):
            hits = hits + 1
        
        if ((searchscreen_actual == 0) & (mem_df_3.searchscreen.iloc[i] == 'low')):
            hits = hits + 1
        hit_list.append(hits)

    mem_df_3['hits'] = hit_list
    mem_df_3['hit_acc'] = mem_df_3['hits']/7

    mem_df_3['hit_acc01'] = [1 if x == 1 else 0 for x in mem_df_3['hit_acc']]

    print(dataset_name)

    # print('For '+ str(k) + ' SI instances in database and '+ sim_type + ' similarity;')
    print('Avg Accuracy per utterance:', mean(mem_df_3['hit_acc']))
    
    return mem_df_3

    


# In[27]:


# #hard
# print('hard membership')
# #cosine
# print('cosine')

# mem_hard_cosine_0 = data_mem_acc('SNIPS_mem_hard_cosine_0.csv')
# mem_hard_cosine_1 = data_mem_acc('SNIPS_mem_hard_cosine_1.csv')
# mem_hard_cosine_2 = data_mem_acc('SNIPS_mem_hard_cosine_2.csv')
# mem_hard_cosine_3 = data_mem_acc('SNIPS_mem_hard_cosine_3.csv')
# mem_hard_cosine_4 = data_mem_acc('SNIPS_mem_hard_cosine_4.csv')

# #jaccard
# print('jaccard')

# mem_hard_jaccard_0 = data_mem_acc('SNIPS_mem_hard_jaccard_0.csv')
# mem_hard_jaccard_1 = data_mem_acc('SNIPS_mem_hard_jaccard_1.csv')
# mem_hard_jaccard_2 = data_mem_acc('SNIPS_mem_hard_jaccard_2.csv')
# mem_hard_jaccard_3 = data_mem_acc('SNIPS_mem_hard_jaccard_3.csv')
# mem_hard_jaccard_4 = data_mem_acc('SNIPS_mem_hard_jaccard_4.csv')

# #partial
# print('partial')

# mem_hard_partial_0 = data_mem_acc('SNIPS_mem_hard_partial_0.csv')
# mem_hard_partial_1 = data_mem_acc('SNIPS_mem_hard_partial_1.csv')
# mem_hard_partial_2 = data_mem_acc('SNIPS_mem_hard_partial_2.csv')
# mem_hard_partial_3 = data_mem_acc('SNIPS_mem_hard_partial_3.csv')
# mem_hard_partial_4 = data_mem_acc('SNIPS_mem_hard_partial_4.csv')

# #token
# print('token')

# mem_hard_token_0 = data_mem_acc('SNIPS_mem_hard_token_0.csv')
# mem_hard_token_1 = data_mem_acc('SNIPS_mem_hard_token_1.csv')
# mem_hard_token_2 = data_mem_acc('SNIPS_mem_hard_token_2.csv')
# mem_hard_token_3 = data_mem_acc('SNIPS_mem_hard_token_3.csv')
# mem_hard_token_4 = data_mem_acc('SNIPS_mem_hard_token_4.csv')


# In[ ]:


#soft
print('soft membership')
#cosine
print('cosine')

mem_soft_cosine_0 = data_mem_acc('SNIPS_mem_soft_cosine_0.csv')
mem_soft_cosine_1 = data_mem_acc('SNIPS_mem_soft_cosine_1.csv')
mem_soft_cosine_2 = data_mem_acc('SNIPS_mem_soft_cosine_2.csv')
mem_soft_cosine_3 = data_mem_acc('SNIPS_mem_soft_cosine_3.csv')
mem_soft_cosine_4 = data_mem_acc('SNIPS_mem_soft_cosine_4.csv')

#jaccard
print('jaccard')

mem_soft_jaccard_0 = data_mem_acc('SNIPS_mem_soft_jaccard_0.csv')
mem_soft_jaccard_1 = data_mem_acc('SNIPS_mem_soft_jaccard_1.csv')
mem_soft_jaccard_2 = data_mem_acc('SNIPS_mem_soft_jaccard_2.csv')
mem_soft_jaccard_3 = data_mem_acc('SNIPS_mem_soft_jaccard_3.csv')
mem_soft_jaccard_4 = data_mem_acc('SNIPS_mem_soft_jaccard_4.csv')

#partial
print('partial')

mem_soft_partial_0 = data_mem_acc('SNIPS_mem_soft_partial_0.csv')
mem_soft_partial_1 = data_mem_acc('SNIPS_mem_soft_partial_1.csv')
mem_soft_partial_2 = data_mem_acc('SNIPS_mem_soft_partial_2.csv')
mem_soft_partial_3 = data_mem_acc('SNIPS_mem_soft_partial_3.csv')
mem_soft_partial_4 = data_mem_acc('SNIPS_mem_soft_partial_4.csv')

#token
print('token')

mem_soft_token_0 = data_mem_acc('SNIPS_mem_soft_token_0.csv')
mem_soft_token_1 = data_mem_acc('SNIPS_mem_soft_token_1.csv')
mem_soft_token_2 = data_mem_acc('SNIPS_mem_soft_token_2.csv')
mem_soft_token_3 = data_mem_acc('SNIPS_mem_soft_token_3.csv')
mem_soft_token_4 = data_mem_acc('SNIPS_mem_soft_token_4.csv')


# In[ ]:


import os

os.chdir('/media/disk4/fuzzy_intents/results/mix_mem_data/mix_mem_final/final_mem/')


# In[ ]:


# #converting membership data to csv

# #hard
# #cosine
# mem_hard_cosine_0.to_csv('SNIPS_mem_hard_cosine_0_r.csv')
# mem_hard_cosine_1.to_csv('SNIPS_mem_hard_cosine_1_r.csv')
# mem_hard_cosine_2.to_csv('SNIPS_mem_hard_cosine_2_r.csv')
# mem_hard_cosine_3.to_csv('SNIPS_mem_hard_cosine_3_r.csv')
# mem_hard_cosine_4.to_csv('SNIPS_mem_hard_cosine_4_r.csv')

# #jaccard
# mem_hard_jaccard_0.to_csv('SNIPS_mem_hard_jaccard_0_r.csv')
# mem_hard_jaccard_1.to_csv('SNIPS_mem_hard_jaccard_1_r.csv')
# mem_hard_jaccard_2.to_csv('SNIPS_mem_hard_jaccard_2_r.csv')
# mem_hard_jaccard_3.to_csv('SNIPS_mem_hard_jaccard_3_r.csv')
# mem_hard_jaccard_4.to_csv('SNIPS_mem_hard_jaccard_4_r.csv')

# #partial
# mem_hard_partial_0.to_csv('SNIPS_mem_hard_partial_0_r.csv')
# mem_hard_partial_1.to_csv('SNIPS_mem_hard_partial_1_r.csv')
# mem_hard_partial_2.to_csv('SNIPS_mem_hard_partial_2_r.csv')
# mem_hard_partial_3.to_csv('SNIPS_mem_hard_partial_3_r.csv')
# mem_hard_partial_4.to_csv('SNIPS_mem_hard_partial_4_r.csv')

# #token
# mem_hard_token_0.to_csv('SNIPS_mem_hard_token_0_r.csv')
# mem_hard_token_1.to_csv('SNIPS_mem_hard_token_1_r.csv')
# mem_hard_token_2.to_csv('SNIPS_mem_hard_token_2_r.csv')
# mem_hard_token_3.to_csv('SNIPS_mem_hard_token_3_r.csv')
# mem_hard_token_4.to_csv('SNIPS_mem_hard_token_4_r.csv')

# ##################################################################

#soft
#cosine
mem_soft_cosine_0.to_csv('SNIPS_mem_soft_cosine_0_r.csv')
mem_soft_cosine_1.to_csv('SNIPS_mem_soft_cosine_1_r.csv')
mem_soft_cosine_2.to_csv('SNIPS_mem_soft_cosine_2_r.csv')
mem_soft_cosine_3.to_csv('SNIPS_mem_soft_cosine_3_r.csv')
mem_soft_cosine_4.to_csv('SNIPS_mem_soft_cosine_4_r.csv')

#jaccard
mem_soft_jaccard_0.to_csv('SNIPS_mem_soft_jaccard_0_r.csv')
mem_soft_jaccard_1.to_csv('SNIPS_mem_soft_jaccard_1_r.csv')
mem_soft_jaccard_2.to_csv('SNIPS_mem_soft_jaccard_2_r.csv')
mem_soft_jaccard_3.to_csv('SNIPS_mem_soft_jaccard_3_r.csv')
mem_soft_jaccard_4.to_csv('SNIPS_mem_soft_jaccard_4_r.csv')

#partial
mem_soft_partial_0.to_csv('SNIPS_mem_soft_partial_0_r.csv')
mem_soft_partial_1.to_csv('SNIPS_mem_soft_partial_1_r.csv')
mem_soft_partial_2.to_csv('SNIPS_mem_soft_partial_2_r.csv')
mem_soft_partial_3.to_csv('SNIPS_mem_soft_partial_3_r.csv')
mem_soft_partial_4.to_csv('SNIPS_mem_soft_partial_4_r.csv')

#token
mem_soft_token_0.to_csv('SNIPS_mem_soft_token_0_r.csv')
mem_soft_token_1.to_csv('SNIPS_mem_soft_token_1_r.csv')
mem_soft_token_2.to_csv('SNIPS_mem_soft_token_2_r.csv')
mem_soft_token_3.to_csv('SNIPS_mem_soft_token_3_r.csv')
mem_soft_token_4.to_csv('SNIPS_mem_soft_token_4_r.csv')


# In[ ]:


print('done!')

