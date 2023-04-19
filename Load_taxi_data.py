# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:36:22 2021

@author: akovacevic
"""
import pandas as pd
import numpy as np

from Prepare_Data import remap_answers_tx
from Prepare_Data import get_aggregated_data
from Prepare_Data import create_ratings_and_mapping
from Prepare_Data import get_user_ids_from_mapping

def Load_TX_Data(expert_type):
    """
    The procedure is used to read taxi data. It reads and cleans data for crowd, 
    driver expert and traffic expert.
    
    Return:
        Dictionary of question_map - dictionary of question names and question ids,
                      alt_names - list of alternative ids,
                      df_crowd - trans data frame of crowd answers,
                      df_selected_expert - trans data frame of selected expert answers,
                      df_driver - trans data frame of drivers answers,
                      df_expert - trans data frame of traffic expert answers,
                      all_votes - trans data frame of all users' answers (crowd, traffic experts, and drivers),
                      expert_ids - ids of expert users,
                      crowd_ids - ids of crowd users,
                      voter_map - id name map of all voters (expert and crowd),
                      df_alt_votes - data frame where alternatives are represented in rows and
                                      users in columns
    Input:
        parameter expert_type (string value) that could take the following values: 
            traffic, driver, all
    """
    ### Load Data
    exp = pd.read_excel(open('TX_data/BG/Eksperti_Anketa_BG.xlsx', 'rb'), sheet_name='OdgovoriIzUpitnika')  
    drv = pd.read_excel(open('TX_data/BG/Vozaci_Anketa_BG.xlsx', 'rb'),  sheet_name='OdgovoriIzUpitnika')
    crd = pd.read_excel(open('TX_data/BG/Korisnici_Anketa_BG.xlsx', 'rb'), sheet_name='OdgovoriIzUpitnika')
    
    ### List of questions needed for analysis
    questions = ['Карактеристике возила [Тип/каросерија возила]',
           'Карактеристике возила [Димензије (ширина врата, гепек...)]',
           'Карактеристике возила [Лак улазак/излазак]',
           'Комфор у возилу [Удобност седишта]',
           'Комфор у возилу [Климатизација и грејање]',
           'Комфор у возилу [Чистоћа возила (спољашњост и унутрашњост)]',
           'Комуникациона опрема [Навигациона мапа (ГПС)]',
           'Комуникациона опрема [Флексибилно плаћање (новац, картица)]',
           'Комуникациона опрема [Опрема за резервацију вожњи (апликација, радио веза, ...)]',
           'Безбедност и дизајн [Старост возила]',
           'Безбедност и дизајн [Опрема у возилу (airbag,АБС ...)]',
           'Безбедност и дизајн [Тип/марка и боја возила]',
           'Еколошка подобност [Ниво буке]',
           'Еколошка подобност [Ниво  аерозагађења]',
           'Еколошка подобност [Чиста погонска енергија]']
    
    #### dict for renaming questions of crowd questionnaire
    crd_rename_dict = { '15.1.Карактеристике возила [Тип/каросерија возила]': 'Карактеристике возила [Тип/каросерија возила]',
           '15.2.Карактеристике возила [Димензије (ширина врата, гепек...)]' : 'Карактеристике возила [Димензије (ширина врата, гепек...)]',
           '15.3.Карактеристике возила [Лак улазак/излазак]' : 'Карактеристике возила [Лак улазак/излазак]',
           '15.4.Комфор у возилу [Удобност седишта]' : 'Комфор у возилу [Удобност седишта]',
           '15.5.Комфор у возилу [Климатизација и грејање]' : 'Комфор у возилу [Климатизација и грејање]',
           '15.6.Комфор у возилу [Чистоћа возила (спољашњост и унутрашњост)]' : 'Комфор у возилу [Чистоћа возила (спољашњост и унутрашњост)]',
           '15.7.Комуникациона опрема [Навигациона мапа (ГПС)]': 'Комуникациона опрема [Навигациона мапа (ГПС)]',
           '15.8.Комуникациона опрема [Прикључак за мобилни телефон]' : 'Комуникациона опрема [Опрема за резервацију вожњи (апликација, радио веза, ...)]',
           '15.9.Комуникациона опрема [Флексибилно плаћање (новац, картица)]' : 'Комуникациона опрема [Флексибилно плаћање (новац, картица)]',
           '15.10.Безбедност и дизајн [Старост возила]': 'Безбедност и дизајн [Старост возила]',
           '15.11.Безбедност и дизајн [Опрема у возилу (појас, airbag, ...)]' : 'Безбедност и дизајн [Опрема у возилу (airbag,АБС ...)]',
           '15.2.Безбедност и дизајн [Тип/марка и боја возила]': 'Безбедност и дизајн [Тип/марка и боја возила]',
           '15.13.Еколошка подобност [Ниво буке]': 'Еколошка подобност [Ниво буке]',
           '15.14.Еколошка подобност [Ниво  аерозагађења]' : 'Еколошка подобност [Ниво  аерозагађења]',
           '15.15.Еколошка подобност [Чиста погонска енергија]' : 'Еколошка подобност [Чиста погонска енергија]'}
    
    ## dict for renaming questions of driver questionnaire 
    driver_rename_dct = {  '2.7.1.1.Pristup u vozilo/Tip,karosterija': 'Карактеристике возила [Тип/каросерија возила]',
           '2.7.1.2.Pristup u vozilo/Dimenzije' : 'Карактеристике возила [Димензије (ширина врата, гепек...)]',
           '2.7.1.3.Pristup u vozilo/Lak Ulazak izlazak' : 'Карактеристике возила [Лак улазак/излазак]',
           '2.7.2.1.Komfor u vozilu/Udobnost sedišta' : 'Комфор у возилу [Удобност седишта]',
           '2.7.2.2.Komfor u vozilu/Klimatizacija i grejanje' : 'Комфор у возилу [Климатизација и грејање]',
           '2.7.2.3.Komfor u vozilu/Čistoća vozila' : 'Комфор у возилу [Чистоћа возила (спољашњост и унутрашњост)]',
           '2.7.3.3.Komunikaciona oprema/GPS' : 'Комуникациона опрема [Навигациона мапа (ГПС)]',
           '2.7.3.1.Komunikaciona oprema/Aplikacija' : 'Комуникациона опрема [Опрема за резервацију вожњи (апликација, радио веза, ...)]',
           '2.7.3.2.Komunikaciona oprema/Radio veza' : 'Комуникациона опрема [Флексибилно плаћање (новац, картица)]',
           '2.7.4.1.Bezbednost i dizajn/Starost vozila' : 'Безбедност и дизајн [Старост возила]',
           '2.7.4.2.Bezbednost i dizajn/Starost vozila' : 'Безбедност и дизајн [Опрема у возилу (airbag,АБС ...)]',
           '2.7.4.3.Bezbednost i dizajn/Tip, marka i boja' : 'Безбедност и дизајн [Тип/марка и боја возила]',
           '2.7.5.1.Ekološka podobnost/Nivo buke' : 'Еколошка подобност [Ниво буке]',
           '2.7.5.2.Ekološka podobnost/Nivo aerozagadjenja' : 'Еколошка подобност [Ниво  аерозагађења]',
           '2.7.5.3.Ekološka podobnost/Čista pogonska energija' : 'Еколошка подобност [Чиста погонска енергија]'}
      
    ### remap textual answers to numbers
    exp = remap_answers_tx(exp)
    crd = remap_answers_tx(crd)
    
    ### rename colums 
    drv = drv.rename(columns=driver_rename_dct)
    crd = crd.rename(columns = crd_rename_dict)
    
    ### select only attributes of interest
    e = exp[questions]
    d = drv[questions]
    c = crd[questions]
    

    #### create id of every user
    e['id'] = e.index
    d['id'] = d.index
    c['id'] = c.index
    
    #### create transactional data
    exp_trans = pd.melt(e, id_vars= ['id'], value_vars= questions,
                        var_name='question', value_name= 'rate')
    
    drv_trans = pd.melt(d, id_vars=['id'], value_vars=questions,
                        var_name='question', value_name='rate')
    
    crd_trans = pd.melt(c, id_vars= ['id'], value_vars= questions,
                        var_name='question', value_name= 'rate')
    
    #### create id for each question
    exp_trans['question_id'] = exp_trans.groupby('question').ngroup()
    
    #### create question map (id to original text)
    question_map = exp_trans[['question_id', 'question']].drop_duplicates().reset_index().drop('index', axis = 1)
    #pd.Series(question_map.question_id.values,index=question_map.question).to_dict()
    drv_trans = pd.merge(drv_trans, question_map, on = 'question')
    drv_trans['rate'] = pd.to_numeric(drv_trans['rate'], errors = 'coerce')
    
    crd_trans = pd.merge(crd_trans, question_map, on= 'question')
    crd_trans['rate'] = pd.to_numeric(crd_trans['rate'], errors = 'coerce')
    
    ### get all alternative names (ids)
    alt_names = list(question_map['question_id'].sort_values())
    
    # create user names (with specified types of each voter)
    exp_trans['voter'] = exp_trans.apply( lambda x: 'traffic_' + str(x['id']) + '_expert', axis = 1)
    drv_trans['voter'] = drv_trans.apply( lambda x: 'driver_' + str(x['id']) + '_expert', axis = 1)
    crd_trans['voter'] = crd_trans.apply( lambda x: str(x['id']) + '_crowd', axis = 1)
    
    #### select attributes for analysis
    df_expert = exp_trans[['voter', 'question_id', 'rate']]
    df_crowd = crd_trans[['voter', 'question_id', 'rate']]
    df_driver = drv_trans[['voter', 'question_id', 'rate']]
    
    df_expert['rate']= df_expert['rate'].astype('float')
    df_crowd['rate']= df_crowd['rate'].astype('float')
    df_driver['rate'] = df_driver['rate'].astype('float')
    
    df_crowd = df_crowd.dropna()
    df_expert = df_expert.dropna()
    df_driver = df_driver.dropna()
    
    ## filter possible errors and remove missing values
    df_expert = df_expert.loc[(df_expert['rate']<=3) & (df_expert['rate']>0)]    
    df_driver = df_driver.loc[(df_driver['rate']<=3) & (df_driver['rate']>0)]
    df_crowd = df_crowd.loc[(df_crowd['rate']<=3) & (df_crowd['rate']>0)] 
    
    all_votes = df_crowd.append(df_expert).append(df_driver)
    
    if expert_type == 'driver':
        df_selected_expert = df_driver
    elif expert_type == 'traffic':
        df_selected_expert = df_expert
    else:
          df_selected_expert = pd.concat([df_expert, df_driver], ignore_index=True)


    df_expert_crowd = pd.concat([df_selected_expert, df_crowd], ignore_index=True)
    #n_crowd = len(df_crowd['voter'].unique())
    
    ############# Aggregate data
    crowd_agg = get_aggregated_data(df_crowd, alt_names, index_column = 'voter', column= 'question_id', value = 'rate')
    expert_agg = get_aggregated_data(df_selected_expert, alt_names, index_column = 'voter', column= 'question_id', value = 'rate')
    expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names, index_column = 'voter', column= 'question_id', value = 'rate')
    
    ############ Create user mapping
    _, _, voter_map = create_ratings_and_mapping(expert_crowd_agg, alt_names, voter_col = 'voter')
    
    ##### replace voters name with ids in all dataframes
    df_crowd = pd.merge( voter_map,df_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
    df_expert_crowd = pd.merge( voter_map, df_expert_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
    df_selected_expert = pd.merge(voter_map,  df_selected_expert, how = 'inner', on = 'voter').drop('voter', axis = 1)
    
    crowd_agg = pd.merge(voter_map,  crowd_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)
    cr_voter = crowd_agg['voter_id']
    crowd_agg = crowd_agg[alt_names].replace(0, np.nan)
    crowd_agg['voter_id'] = cr_voter
    
    expert_agg = pd.merge(voter_map,  expert_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)
    exp_voter = expert_agg['voter_id']
    expert_agg = expert_agg[alt_names].replace(0, np.nan)
    expert_agg['voter_id'] = exp_voter
    
    #### extract expert and crowd ids for similarity
    expert_ids = get_user_ids_from_mapping(voter_map, 'expert')
    crowd_ids = get_user_ids_from_mapping(voter_map, 'crowd')
    
    ### create data set where alternatives represents rows and voterids are in colums
    df_alt_votes = get_aggregated_data(pd.concat([df_crowd, df_selected_expert]), voter_map['voter_id'], 
                                   index_column = 'question_id', column= 'voter_id', value = 'rate')
    qu = df_alt_votes['question_id']
    df_alt_votes = df_alt_votes[crowd_ids + expert_ids].replace(0, np.nan)
    df_alt_votes['question_id'] = qu
    
    result_dict = {'question_map' : question_map,
                   'alt_names': alt_names,
                   'df_crowd' : df_crowd,
                   'df_selected_expert' : df_selected_expert,
                   'df_driver': df_driver,
                   'df_traffic': df_expert,
                   'all_votes' : all_votes,
                   'expert_ids' : expert_ids,
                   'crowd_ids' : crowd_ids,
                   'voter_map' : voter_map,
                   'df_alt_votes' : df_alt_votes}

    return result_dict
