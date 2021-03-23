# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:36:22 2021

@author: akovacevic
"""
import pandas as pd

from Data_Prepare import remap_answers_tx

def Load_Data():
    
    exp = pd.read_excel(open('TX_data/BG/Eksperti_Anketa_BG.xlsx', 'rb'), sheet_name='OdgovoriIzUpitnika')  
    crd = pd.read_excel(open('TX_data/BG/Korisnici_Anketa_BG.xlsx', 'rb'), sheet_name='OdgovoriIzUpitnika')
    
    
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
    
    rename_dict = { '15.1.Карактеристике возила [Тип/каросерија возила]': 'Карактеристике возила [Тип/каросерија возила]',
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
    
    exp = remap_answers_tx(exp)
    crd = remap_answers_tx(crd)
    crd = crd.rename(columns = rename_dict)
    e = exp[questions]
    c = crd[questions]
    
    e['id'] = e.index
    c['id'] = c.index
    
    
    exp_trans = pd.melt(e, id_vars= ['id'], value_vars= questions,
                        var_name='question', value_name= 'rate')
    
    exp_trans['question_id'] = exp_trans.groupby('question').ngroup()
    question_map = exp_trans[['question_id', 'question']].drop_duplicates().reset_index().drop('index', axis = 1)
    alt_names = list(question_map['question_id'].sort_values())
    
    crd_trans = pd.melt(c, id_vars= ['id'], value_vars= questions,
                        var_name='question', value_name= 'rate')
    
    crd_trans = pd.merge(crd_trans, question_map, on= 'question')
    crd_trans['rate'] = pd.to_numeric(crd_trans['rate'], errors = 'coerce')


    exp_trans['voter'] = exp_trans.apply( lambda x: 'traffic_' + str(x['id']) + '_expert', axis = 1)
    crd_trans['voter'] = crd_trans.apply( lambda x: str(x['id']) + '_crowd', axis = 1)
    
    
    #alternative_map, alt_names, df_crowd, _, _ , df_science, df_journal = read_data_credibility()
    df_expert = exp_trans[['voter', 'question_id', 'rate']]
    df_crowd = crd_trans[['voter', 'question_id', 'rate']]
    
    df_expert['rate']= df_expert['rate'].astype('float')
    df_crowd['rate']= df_crowd['rate'].astype('float')
    
    df_crowd.loc[df_crowd['rate']>3, 'rate'] = 3
    
    
    df_crowd = df_crowd.dropna()
    df_expert = df_expert.dropna()
    
    return question_map, alt_names, df_crowd, df_expert 
