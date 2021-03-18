#!/usr/bin/env python
# coding: utf-8

# In[14]:


#softmax mem data
import pandas as pd

softmax_mem = pd.read_csv('./results/softmax_data/SI_softmax_SNIPS.csv')
softmax_mem = softmax_mem.iloc[:,1:]
softmax_mem = softmax_mem.drop_duplicates()


# In[15]:


#hardmax mem data
hardmax_mem = pd.read_csv('./results/softmax_data/SI_hardmax_SNIPS.csv')
hardmax_mem = hardmax_mem.iloc[:,1:]
hardmax_mem = hardmax_mem.drop_duplicates()


# In[16]:


token_df0 = pd.read_csv('./results/sim_data/tokenset_ratio_map_SNIPS_0.csv')
partial_df0 = pd.read_csv('./results/sim_data/partial_ratio_map_SNIPS_0.csv')
jaccard_df0 = pd.read_csv('./results/sim_data/jaccard_ratio_map_SNIPS_0.csv')
cosine_df0 = pd.read_csv('./results/sim_data/cosine_ratio_map_SNIPS_0.csv')

token_df1 = pd.read_csv('./results/sim_data/tokenset_ratio_map_SNIPS_1.csv')
partial_df1 = pd.read_csv('./results/sim_data/partial_ratio_map_SNIPS_1.csv')
jaccard_df1 = pd.read_csv('./results/sim_data/jaccard_ratio_map_SNIPS_1.csv')
cosine_df1 = pd.read_csv('./results/sim_data/cosine_ratio_map_SNIPS_1.csv')

token_df2 = pd.read_csv('./results/sim_data/tokenset_ratio_map_SNIPS_2.csv')
partial_df2 = pd.read_csv('./results/sim_data/partial_ratio_map_SNIPS_2.csv')
jaccard_df2 = pd.read_csv('./results/sim_data/jaccard_ratio_map_SNIPS_2.csv')
cosine_df2 = pd.read_csv('./results/sim_data/cosine_ratio_map_SNIPS_2.csv')

token_df3 = pd.read_csv('./results/sim_data/tokenset_ratio_map_SNIPS_3.csv')
partial_df3 = pd.read_csv('./results/sim_data/partial_ratio_map_SNIPS_3.csv')
jaccard_df3 = pd.read_csv('./results/sim_data/jaccard_ratio_map_SNIPS_3.csv')
cosine_df3 = pd.read_csv('./results/sim_data/cosine_ratio_map_SNIPS_3.csv')

token_df4 = pd.read_csv('./results/sim_data/tokenset_ratio_map_SNIPS_4.csv')
partial_df4 = pd.read_csv('./results/sim_data/partial_ratio_map_SNIPS_4.csv')
jaccard_df4 = pd.read_csv('./results/sim_data/jaccard_ratio_map_SNIPS_4.csv')
cosine_df4 = pd.read_csv('./results/sim_data/cosine_ratio_map_SNIPS_4.csv')


# In[17]:


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


# In[18]:


token_df0 = sim_setup(token_df0)
token_df1 = sim_setup(token_df1)
token_df2 = sim_setup(token_df2)
token_df3 = sim_setup(token_df3)
token_df4 = sim_setup(token_df4)

partial_df0 = sim_setup(partial_df0)
partial_df1 = sim_setup(partial_df1)
partial_df2 = sim_setup(partial_df2)
partial_df3 = sim_setup(partial_df3)
partial_df4 = sim_setup(partial_df4)

cosine_df0 = sim_setup(cosine_df0)
cosine_df1 = sim_setup(cosine_df1)
cosine_df2 = sim_setup(cosine_df2)
cosine_df3 = sim_setup(cosine_df3)
cosine_df4 = sim_setup(cosine_df4)

jaccard_df0 = sim_setup(jaccard_df0)
jaccard_df1 = sim_setup(jaccard_df1)
jaccard_df2 = sim_setup(jaccard_df2)
jaccard_df3 = sim_setup(jaccard_df3)
jaccard_df4 = sim_setup(jaccard_df4)


# In[19]:


def member_map_mi(sim_data_type, softmax_data):
    
    sim_data = sim_data_type
    softmax_mem = softmax_data
    
    mix_list = []
    si_list = []
    intent_list = []
    score_list = []
    mem_list = []
    AddToPlaylist_l_list = []
    AddToPlaylist_m_list = []
    AddToPlaylist_h_list = []
    BookRestaurant_l_list = []
    BookRestaurant_m_list = []
    BookRestaurant_h_list = []
    GetWeather_l_list = []
    GetWeather_m_list = []
    GetWeather_h_list = []
    PlayMusic_l_list = []
    PlayMusic_m_list = []
    PlayMusic_h_list = []
    RateBook_l_list = []
    RateBook_m_list = []
    RateBook_h_list = []
    SearchCreativeWork_l_list = []
    SearchCreativeWork_m_list = []
    SearchCreativeWork_h_list = []
    SearchScreeningEvent_l_list = []
    SearchScreeningEvent_m_list = []
    SearchScreeningEvent_h_list = []


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
            for j in range(len(softmax_data.utterance)):
                si = softmax_data.utterance.iloc[j]
                if sent == si:
                    mix_list.append(utt)
                    si_list.append(sent)
                    intent_list.append(intent1)
                    score_list.append(score1)
                    AddToPlaylist_l_list.append(softmax_data.AddToPlaylist_low.iloc[j])
                    AddToPlaylist_m_list.append(softmax_data.AddToPlaylist_med.iloc[j])
                    AddToPlaylist_h_list.append(softmax_data.AddToPlaylist_high.iloc[j])
                    BookRestaurant_l_list.append(softmax_data.BookRestaurant_low.iloc[j])
                    BookRestaurant_m_list.append(softmax_data.BookRestaurant_med.iloc[j])
                    BookRestaurant_h_list.append(softmax_data.BookRestaurant_high.iloc[j])
                    GetWeather_l_list.append(softmax_data.GetWeather_low.iloc[j])
                    GetWeather_m_list.append(softmax_data.GetWeather_med.iloc[j])
                    GetWeather_h_list.append(softmax_data.GetWeather_high.iloc[j])
                    PlayMusic_l_list.append(softmax_data.PlayMusic_low.iloc[j])
                    PlayMusic_m_list.append(softmax_data.PlayMusic_med.iloc[j])
                    PlayMusic_h_list.append(softmax_data.PlayMusic_high.iloc[j])
                    RateBook_l_list.append(softmax_data.RateBook_low.iloc[j])
                    RateBook_m_list.append(softmax_data.RateBook_med.iloc[j])
                    RateBook_h_list.append(softmax_data.RateBook_high.iloc[j])
                    SearchCreativeWork_l_list.append(softmax_data.SearchCreativeWork_low.iloc[j])
                    SearchCreativeWork_m_list.append(softmax_data.SearchCreativeWork_med.iloc[j])
                    SearchCreativeWork_h_list.append(softmax_data.SearchCreativeWork_high.iloc[j])
                    SearchScreeningEvent_l_list.append(softmax_data.SearchScreeningEvent_low.iloc[j])
                    SearchScreeningEvent_m_list.append(softmax_data.SearchScreeningEvent_med.iloc[j])
                    SearchScreeningEvent_h_list.append(softmax_data.SearchScreeningEvent_high.iloc[j])

        for sent in sent2:
            for j in range(len(softmax_data.utterance)):
                si = softmax_data.utterance.iloc[j]
                if sent == si:
                    mix_list.append(utt)
                    si_list.append(sent)
                    intent_list.append(intent2)
                    score_list.append(score2)
                    AddToPlaylist_l_list.append(softmax_data.AddToPlaylist_low.iloc[j])
                    AddToPlaylist_m_list.append(softmax_data.AddToPlaylist_med.iloc[j])
                    AddToPlaylist_h_list.append(softmax_data.AddToPlaylist_high.iloc[j])
                    BookRestaurant_l_list.append(softmax_data.BookRestaurant_low.iloc[j])
                    BookRestaurant_m_list.append(softmax_data.BookRestaurant_med.iloc[j])
                    BookRestaurant_h_list.append(softmax_data.BookRestaurant_high.iloc[j])
                    GetWeather_l_list.append(softmax_data.GetWeather_low.iloc[j])
                    GetWeather_m_list.append(softmax_data.GetWeather_med.iloc[j])
                    GetWeather_h_list.append(softmax_data.GetWeather_high.iloc[j])
                    PlayMusic_l_list.append(softmax_data.PlayMusic_low.iloc[j])
                    PlayMusic_m_list.append(softmax_data.PlayMusic_med.iloc[j])
                    PlayMusic_h_list.append(softmax_data.PlayMusic_high.iloc[j])
                    RateBook_l_list.append(softmax_data.RateBook_low.iloc[j])
                    RateBook_m_list.append(softmax_data.RateBook_med.iloc[j])
                    RateBook_h_list.append(softmax_data.RateBook_high.iloc[j])
                    SearchCreativeWork_l_list.append(softmax_data.SearchCreativeWork_low.iloc[j])
                    SearchCreativeWork_m_list.append(softmax_data.SearchCreativeWork_med.iloc[j])
                    SearchCreativeWork_h_list.append(softmax_data.SearchCreativeWork_high.iloc[j])
                    SearchScreeningEvent_l_list.append(softmax_data.SearchScreeningEvent_low.iloc[j])
                    SearchScreeningEvent_m_list.append(softmax_data.SearchScreeningEvent_med.iloc[j])
                    SearchScreeningEvent_h_list.append(softmax_data.SearchScreeningEvent_high.iloc[j])

        for sent in sent3:
            for j in range(len(softmax_data.utterance)):
                si = softmax_data.utterance.iloc[j]
                if sent == si:
                    mix_list.append(utt)
                    si_list.append(sent)
                    intent_list.append(intent3)
                    score_list.append(score3)
                    AddToPlaylist_l_list.append(softmax_data.AddToPlaylist_low.iloc[j])
                    AddToPlaylist_m_list.append(softmax_data.AddToPlaylist_med.iloc[j])
                    AddToPlaylist_h_list.append(softmax_data.AddToPlaylist_high.iloc[j])
                    BookRestaurant_l_list.append(softmax_data.BookRestaurant_low.iloc[j])
                    BookRestaurant_m_list.append(softmax_data.BookRestaurant_med.iloc[j])
                    BookRestaurant_h_list.append(softmax_data.BookRestaurant_high.iloc[j])
                    GetWeather_l_list.append(softmax_data.GetWeather_low.iloc[j])
                    GetWeather_m_list.append(softmax_data.GetWeather_med.iloc[j])
                    GetWeather_h_list.append(softmax_data.GetWeather_high.iloc[j])
                    PlayMusic_l_list.append(softmax_data.PlayMusic_low.iloc[j])
                    PlayMusic_m_list.append(softmax_data.PlayMusic_med.iloc[j])
                    PlayMusic_h_list.append(softmax_data.PlayMusic_high.iloc[j])
                    RateBook_l_list.append(softmax_data.RateBook_low.iloc[j])
                    RateBook_m_list.append(softmax_data.RateBook_med.iloc[j])
                    RateBook_h_list.append(softmax_data.RateBook_high.iloc[j])
                    SearchCreativeWork_l_list.append(softmax_data.SearchCreativeWork_low.iloc[j])
                    SearchCreativeWork_m_list.append(softmax_data.SearchCreativeWork_med.iloc[j])
                    SearchCreativeWork_h_list.append(softmax_data.SearchCreativeWork_high.iloc[j])
                    SearchScreeningEvent_l_list.append(softmax_data.SearchScreeningEvent_low.iloc[j])
                    SearchScreeningEvent_m_list.append(softmax_data.SearchScreeningEvent_med.iloc[j])
                    SearchScreeningEvent_h_list.append(softmax_data.SearchScreeningEvent_high.iloc[j])
                    
        multi_mem_df = pd.DataFrame(zip(mix_list, si_list, intent_list, score_list, AddToPlaylist_l_list, AddToPlaylist_m_list,
                                AddToPlaylist_h_list, BookRestaurant_l_list, BookRestaurant_m_list, 
                                BookRestaurant_h_list, GetWeather_l_list, GetWeather_m_list, GetWeather_h_list, 
                                PlayMusic_l_list, PlayMusic_m_list, PlayMusic_h_list, RateBook_l_list,
                               RateBook_m_list, RateBook_h_list, SearchCreativeWork_l_list, SearchCreativeWork_m_list,
                               SearchCreativeWork_h_list, SearchScreeningEvent_l_list, SearchScreeningEvent_m_list,
                               SearchScreeningEvent_h_list), columns = ['multi', 'single', 'intent', 'sim_score', 
                                                                 'addtoplay_l', 'addtoplay_m', 'addtoplay_h',
                                                                'bookrestro_l','bookrestro_m', 'bookrestro_h', 
                                                                'getweath_l', 'getweath_m','getweath_h', 
                                                                'playmusic_l', 'playmusic_m', 'playmusic_h',
                                                                'ratebook_l', 'ratebook_m', 'ratebook_h',
                                                                'searchcreat_l', 'searchcreat_m', 'searchcreat_h',
                                                                'searchscreen_l', 'searchscreen_m', 'searchscreen_h'])
    
    return multi_mem_df
    


# In[20]:


# token_soft_0 = member_map_mi(token_df0,softmax_mem)
# token_hard_0 = member_map_mi(token_df0,hardmax_mem)


# In[ ]:


##softmax 

##tokenset
token_soft_0 = member_map_mi(token_df0,softmax_mem)
token_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_token_0.csv', index = False)

token_soft_1 = member_map_mi(token_df1,softmax_mem)
token_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_token_1.csv', index = False)

token_soft_2 = member_map_mi(token_df2,softmax_mem)
token_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_token_2.csv', index = False)

token_soft_3 = member_map_mi(token_df3,softmax_mem)
token_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_token_3.csv', index = False)

token_soft_4 = member_map_mi(token_df4,softmax_mem)
token_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_token_4.csv', index = False)


##jaccard
jaccard_soft_0 = member_map_mi(jaccard_df0,softmax_mem)
jaccard_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_jaccard_0.csv', index = False)

jaccard_soft_1 = member_map_mi(jaccard_df1,softmax_mem)
jaccard_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_jaccard_1.csv', index = False)

jaccard_soft_2 = member_map_mi(jaccard_df2,softmax_mem)
jaccard_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_jaccard_2.csv', index = False)

jaccard_soft_3 = member_map_mi(jaccard_df3,softmax_mem)
jaccard_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_jaccard_3.csv', index = False)

jaccard_soft_4 = member_map_mi(jaccard_df4,softmax_mem)
jaccard_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_jaccard_4.csv', index = False)

##cosine
cosine_soft_0 = member_map_mi(cosine_df0,softmax_mem)
cosine_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_cosine_0.csv', index = False)

cosine_soft_1 = member_map_mi(cosine_df1,softmax_mem)
cosine_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_cosine_1.csv', index = False)

cosine_soft_2 = member_map_mi(cosine_df2,softmax_mem)
cosine_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_cosine_2.csv', index = False)

cosine_soft_3 = member_map_mi(cosine_df3,softmax_mem)
cosine_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_cosine_3.csv', index = False)

cosine_soft_4 = member_map_mi(cosine_df4,softmax_mem)
cosine_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_cosine_4.csv', index = False)

##partial
partial_soft_0 = member_map_mi(partial_df0,softmax_mem)
partial_soft_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_partial_0.csv', index = False)

partial_soft_1 = member_map_mi(partial_df1,softmax_mem)
partial_soft_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_partial_1.csv', index = False)

partial_soft_2 = member_map_mi(partial_df2,softmax_mem)
partial_soft_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_partial_2.csv', index = False)

partial_soft_3 = member_map_mi(partial_df3,softmax_mem)
partial_soft_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_partial_3.csv', index = False)

partial_soft_4 = member_map_mi(partial_df4,softmax_mem)
partial_soft_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_soft_partial_4.csv', index = False)


# In[ ]:


# ##hardmax

# ##tokenset
# token_hard_0 = member_map_mi(token_df0,hardmax_mem)
# token_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_token_0.csv', index = False)

# token_hard_1 = member_map_mi(token_df1,hardmax_mem)
# token_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_token_1.csv', index = False)

# token_hard_2 = member_map_mi(token_df2,hardmax_mem)
# token_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_token_2.csv', index = False)

# token_hard_3 = member_map_mi(token_df3,hardmax_mem)
# token_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_token_3.csv', index = False)

# token_hard_4 = member_map_mi(token_df4,hardmax_mem)
# token_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_token_4.csv', index = False)


# ##jaccard
# jaccard_hard_0 = member_map_mi(jaccard_df0,hardmax_mem)
# jaccard_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_jaccard_0.csv', index = False)

# jaccard_hard_1 = member_map_mi(jaccard_df1,hardmax_mem)
# jaccard_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_jaccard_1.csv', index = False)

# jaccard_hard_2 = member_map_mi(jaccard_df2,hardmax_mem)
# jaccard_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_jaccard_2.csv', index = False)

# jaccard_hard_3 = member_map_mi(jaccard_df3,hardmax_mem)
# jaccard_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_jaccard_3.csv', index = False)

# jaccard_hard_4 = member_map_mi(jaccard_df4,hardmax_mem)
# jaccard_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_jaccard_4.csv', index = False)

# ##cosine
# cosine_hard_0 = member_map_mi(cosine_df0,hardmax_mem)
# cosine_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_cosine_0.csv', index = False)

# cosine_hard_1 = member_map_mi(cosine_df1,hardmax_mem)
# cosine_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_cosine_1.csv', index = False)

# cosine_hard_2 = member_map_mi(cosine_df2,hardmax_mem)
# cosine_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_cosine_2.csv', index = False)

# cosine_hard_3 = member_map_mi(cosine_df3,hardmax_mem)
# cosine_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_cosine_3.csv', index = False)

# cosine_hard_4 = member_map_mi(cosine_df4,hardmax_mem)
# cosine_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_cosine_4.csv', index = False)

# ##partial
# partial_hard_0 = member_map_mi(partial_df0,hardmax_mem)
# partial_hard_0.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_partial_0.csv', index = False)

# partial_hard_1 = member_map_mi(partial_df1,hardmax_mem)
# partial_hard_1.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_partial_1.csv', index = False)

# partial_hard_2 = member_map_mi(partial_df2,hardmax_mem)
# partial_hard_2.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_partial_2.csv', index = False)

# partial_hard_3 = member_map_mi(partial_df3,hardmax_mem)
# partial_hard_3.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_partial_3.csv', index = False)

# partial_hard_4 = member_map_mi(partial_df4,hardmax_mem)
# partial_hard_4.to_csv('./results/mix_mem_data/mix_mem_final/SNIPS_mem_hard_partial_4.csv', index = False)


# In[5]:


# #map MI to mems

# mix_list = []
# si_list = []
# intent_list = []
# score_list = []
# mem_list = []
# AddToPlaylist_l_list = []
# AddToPlaylist_m_list = []
# AddToPlaylist_h_list = []
# BookRestaurant_l_list = []
# BookRestaurant_m_list = []
# BookRestaurant_h_list = []
# GetWeather_l_list = []
# GetWeather_m_list = []
# GetWeather_h_list = []
# PlayMusic_l_list = []
# PlayMusic_m_list = []
# PlayMusic_h_list = []
# RateBook_l_list = []
# RateBook_m_list = []
# RateBook_h_list = []
# SearchCreativeWork_l_list = []
# SearchCreativeWork_m_list = []
# SearchCreativeWork_h_list = []
# SearchScreeningEvent_l_list = []
# SearchScreeningEvent_m_list = []
# SearchScreeningEvent_h_list = []


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
#         for j in range(len(hardmax_mem.utterance)):
#             si = hardmax_mem.utterance.iloc[j]
#             if sent == si:
#                 mix_list.append(utt)
#                 si_list.append(sent)
#                 intent_list.append(intent1)
#                 score_list.append(score1)
#                 AddToPlaylist_l_list.append(hardmax_mem.AddToPlaylist_low.iloc[j])
#                 AddToPlaylist_m_list.append(hardmax_mem.AddToPlaylist_med.iloc[j])
#                 AddToPlaylist_h_list.append(hardmax_mem.AddToPlaylist_high.iloc[j])
#                 BookRestaurant_l_list.append(hardmax_mem.BookRestaurant_low.iloc[j])
#                 BookRestaurant_m_list.append(hardmax_mem.BookRestaurant_med.iloc[j])
#                 BookRestaurant_h_list.append(hardmax_mem.BookRestaurant_high.iloc[j])
#                 GetWeather_l_list.append(hardmax_mem.GetWeather_low.iloc[j])
#                 GetWeather_m_list.append(hardmax_mem.GetWeather_med.iloc[j])
#                 GetWeather_h_list.append(hardmax_mem.GetWeather_high.iloc[j])
#                 PlayMusic_l_list.append(hardmax_mem.PlayMusic_low.iloc[j])
#                 PlayMusic_m_list.append(hardmax_mem.PlayMusic_med.iloc[j])
#                 PlayMusic_h_list.append(hardmax_mem.PlayMusic_high.iloc[j])
#                 RateBook_l_list.append(hardmax_mem.RateBook_low.iloc[j])
#                 RateBook_m_list.append(hardmax_mem.RateBook_med.iloc[j])
#                 RateBook_h_list.append(hardmax_mem.RateBook_high.iloc[j])
#                 SearchCreativeWork_l_list.append(hardmax_mem.SearchCreativeWork_low.iloc[j])
#                 SearchCreativeWork_m_list.append(hardmax_mem.SearchCreativeWork_med.iloc[j])
#                 SearchCreativeWork_h_list.append(hardmax_mem.SearchCreativeWork_high.iloc[j])
#                 SearchScreeningEvent_l_list.append(hardmax_mem.SearchScreeningEvent_low.iloc[j])
#                 SearchScreeningEvent_m_list.append(hardmax_mem.SearchScreeningEvent_med.iloc[j])
#                 SearchScreeningEvent_h_list.append(hardmax_mem.SearchScreeningEvent_high.iloc[j])

#     for sent in sent2:
#         for j in range(len(hardmax_mem.utterance)):
#             si = hardmax_mem.utterance.iloc[j]
#             if sent == si:
#                 mix_list.append(utt)
#                 si_list.append(sent)
#                 intent_list.append(intent2)
#                 score_list.append(score2)
#                 AddToPlaylist_l_list.append(hardmax_mem.AddToPlaylist_low.iloc[j])
#                 AddToPlaylist_m_list.append(hardmax_mem.AddToPlaylist_med.iloc[j])
#                 AddToPlaylist_h_list.append(hardmax_mem.AddToPlaylist_high.iloc[j])
#                 BookRestaurant_l_list.append(hardmax_mem.BookRestaurant_low.iloc[j])
#                 BookRestaurant_m_list.append(hardmax_mem.BookRestaurant_med.iloc[j])
#                 BookRestaurant_h_list.append(hardmax_mem.BookRestaurant_high.iloc[j])
#                 GetWeather_l_list.append(hardmax_mem.GetWeather_low.iloc[j])
#                 GetWeather_m_list.append(hardmax_mem.GetWeather_med.iloc[j])
#                 GetWeather_h_list.append(hardmax_mem.GetWeather_high.iloc[j])
#                 PlayMusic_l_list.append(hardmax_mem.PlayMusic_low.iloc[j])
#                 PlayMusic_m_list.append(hardmax_mem.PlayMusic_med.iloc[j])
#                 PlayMusic_h_list.append(hardmax_mem.PlayMusic_high.iloc[j])
#                 RateBook_l_list.append(hardmax_mem.RateBook_low.iloc[j])
#                 RateBook_m_list.append(hardmax_mem.RateBook_med.iloc[j])
#                 RateBook_h_list.append(hardmax_mem.RateBook_high.iloc[j])
#                 SearchCreativeWork_l_list.append(hardmax_mem.SearchCreativeWork_low.iloc[j])
#                 SearchCreativeWork_m_list.append(hardmax_mem.SearchCreativeWork_med.iloc[j])
#                 SearchCreativeWork_h_list.append(hardmax_mem.SearchCreativeWork_high.iloc[j])
#                 SearchScreeningEvent_l_list.append(hardmax_mem.SearchScreeningEvent_low.iloc[j])
#                 SearchScreeningEvent_m_list.append(hardmax_mem.SearchScreeningEvent_med.iloc[j])
#                 SearchScreeningEvent_h_list.append(hardmax_mem.SearchScreeningEvent_high.iloc[j])

#     for sent in sent3:
#         for j in range(len(hardmax_mem.utterance)):
#             si = hardmax_mem.utterance.iloc[j]
#             if sent == si:
#                 mix_list.append(utt)
#                 si_list.append(sent)
#                 intent_list.append(intent3)
#                 score_list.append(score3)
#                 AddToPlaylist_l_list.append(hardmax_mem.AddToPlaylist_low.iloc[j])
#                 AddToPlaylist_m_list.append(hardmax_mem.AddToPlaylist_med.iloc[j])
#                 AddToPlaylist_h_list.append(hardmax_mem.AddToPlaylist_high.iloc[j])
#                 BookRestaurant_l_list.append(hardmax_mem.BookRestaurant_low.iloc[j])
#                 BookRestaurant_m_list.append(hardmax_mem.BookRestaurant_med.iloc[j])
#                 BookRestaurant_h_list.append(hardmax_mem.BookRestaurant_high.iloc[j])
#                 GetWeather_l_list.append(hardmax_mem.GetWeather_low.iloc[j])
#                 GetWeather_m_list.append(hardmax_mem.GetWeather_med.iloc[j])
#                 GetWeather_h_list.append(hardmax_mem.GetWeather_high.iloc[j])
#                 PlayMusic_l_list.append(hardmax_mem.PlayMusic_low.iloc[j])
#                 PlayMusic_m_list.append(hardmax_mem.PlayMusic_med.iloc[j])
#                 PlayMusic_h_list.append(hardmax_mem.PlayMusic_high.iloc[j])
#                 RateBook_l_list.append(hardmax_mem.RateBook_low.iloc[j])
#                 RateBook_m_list.append(hardmax_mem.RateBook_med.iloc[j])
#                 RateBook_h_list.append(hardmax_mem.RateBook_high.iloc[j])
#                 SearchCreativeWork_l_list.append(hardmax_mem.SearchCreativeWork_low.iloc[j])
#                 SearchCreativeWork_m_list.append(hardmax_mem.SearchCreativeWork_med.iloc[j])
#                 SearchCreativeWork_h_list.append(hardmax_mem.SearchCreativeWork_high.iloc[j])
#                 SearchScreeningEvent_l_list.append(hardmax_mem.SearchScreeningEvent_low.iloc[j])
#                 SearchScreeningEvent_m_list.append(hardmax_mem.SearchScreeningEvent_med.iloc[j])
#                 SearchScreeningEvent_h_list.append(hardmax_mem.SearchScreeningEvent_high.iloc[j])
                


# In[6]:


# multi_mem_df = pd.DataFrame(zip(mix_list, si_list, intent_list, score_list, AddToPlaylist_l_list, AddToPlaylist_m_list,
#                                 AddToPlaylist_h_list, BookRestaurant_l_list, BookRestaurant_m_list, 
#                                 BookRestaurant_h_list, GetWeather_l_list, GetWeather_m_list, GetWeather_h_list, 
#                                 PlayMusic_l_list, PlayMusic_m_list, PlayMusic_h_list, RateBook_l_list,
#                                RateBook_m_list, RateBook_h_list, SearchCreativeWork_l_list, SearchCreativeWork_m_list,
#                                SearchCreativeWork_h_list, SearchScreeningEvent_l_list, SearchScreeningEvent_m_list,
#                                SearchScreeningEvent_h_list), columns = ['multi', 'single', 'intent', 'sim_score', 
#                                                                  'addtoplay_l', 'addtoplay_m', 'addtoplay_h',
#                                                                 'bookrestro_l','bookrestro_m', 'bookrestro_h', 
#                                                                 'getweath_l', 'getweath_m','getweath_h', 
#                                                                 'playmusic_l', 'playmusic_m', 'playmusic_h',
#                                                                 'ratebook_l', 'ratebook_m', 'ratebook_h',
#                                                                 'searchcreat_l', 'searchcreat_m', 'searchcreat_h',
#                                                                 'searchscreen_l', 'searchscreen_m', 'searchscreen_h'])


# In[33]:


# # multi_mem_df.to_csv('./results/mix_mem_data/mix_mem_2/SNIPS_multi_mem_'+sim_type+'_ratio_'+str(k)+'.csv', index = False)
# multi_mem_df.to_csv('./results/mix_mem_data/mix_mem_1/hard_SNIPS_multi_mem_cosine_ratio_1000.csv', index = False)


# In[ ]:




