import pandas as pd
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer, Metadata, Interpreter
import HTMLParser
import math
import re
import json
import numpy
from nltk import word_tokenize, sent_tokenize, pos_tag
import gensim
from copy import deepcopy
from termcolor import colored
global tag_df
tag_df = pd.concat(pd.read_csv('./taglist_3.csv', chunksize = 10000, compression='gzip'), axis=0)
tag_df['tag'] = ['---' + i.split('---')[1] + '---' for i in tag_df.tags]

global interpreter
interpreter = Interpreter.load('../Codes and More/default/my_model/',
                               RasaNLUConfig("../Codes and More/config_spacy.json"))

global model
model = gensim.models.doc2vec.Doc2Vec.load('./model_3.bin')

global reg
reg = pd.read_csv('../Codes and More/alcohol_regulations_US_2.csv')

global synonym
with open('../Codes and More/default/my_model/entity_synonyms copy.json', 'r') as syn_json :
    synonym = json.load(syn_json)
    syn_json.close()

global states
states = {'AL' : 'Alabama', 'AK' : 'Alaska', 'AZ' : 'Arizona', 'CA' : 'California', 'CO' : 'Colorado',
          'CT' : 'Connecticut', 'DE' : 'Delaware', 'DC' : 'District of Columbia', 'FL' : 'Florida',
          'GA' : 'Georgia', 'HI' : 'Hawaii', 'ID' : 'Idaho', 'IL' : 'Illinois', 'IN' : 'Indiana', 'IA' : 'Iowa',
          'KS' : 'Kansas', 'KY' : 'Kentucky', 'LA' : 'Louisiana', 'ME' : 'Maine', 'MA' : 'Maryland',
          'MI' : 'Michigan', 'MN' : 'Minnesota', 'MS' : 'Mississippi', 'MO' : 'Missouri', 'MT' : 'Montana',
          'NE' : 'Nebraska', 'NV' : 'Nevada', 'NH' : 'New Hampshire', 'NM' : 'New Mexico', 'NY' : 'New York',
          'NC' : 'North Carolina', 'ND' : 'North Dakota', 'OH' : 'Ohio', 'OK' : 'Oklahoma', 'OR' : 'Oregon',
          'SC' : 'South California', 'SD' : 'South Dakota', 'TN' : 'Tennessee', 'TX' : 'Texas', 'UT' : 'Utah',
          'VT' : 'Vermont', 'VA' : 'Virginia', 'WA' : 'Washington', 'WV' : 'West Virginia', 'WI' : 'Wisconsin',
          'WY' : 'Wyoming', 'AR' : 'Arkansas'}
global state_df
state_df = pd.DataFrame({'code' : states.keys(), 'state' : [k.lower() for k in states.values()]})

global params
params = {'question' : '', 'state' : '', 'intent' : '', 'details' : ''}

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def check_answer(question) :
    answer = interpreter.parse(unicode(question).replace('?', ''))
    entities = answer['entities']
    st_ind = [i for i in range(len(entities)) if  entities[i]['entity'] == 'state']
    if (len(st_ind) > 0) :
        if (entities[st_ind[0]]['value'] not in [s.swapcase() for s in states.keys()]) :
            entities.pop(st_ind[0])
    entities = [entities[i].pop('extractor') for i in range(len(entities))]
    return answer

def retrain_using_new_info(answer, intent = 'OK', synonym = {}) :
    entities = answer['entities']
    print intent
    # new_answer = check_answer(answer)
    if intent == 'OK' :
        answer['intent'] = answer['intent']['name']
    else :
        answer['intent'] = intent
    answer.pop('intent_ranking')
    with open('../Codes and More/question_training_data2.json') as json_train :
        train = json.load(json_train)
        json_train.close()
    st_ind = [i for i in range(len(entities)) if  entities[i]['entity'] == 'state']
    if len(st_ind) > 0 :
        curr_state = answer['entities'][st_ind[0]]['value'].lower()
    else :
        curr_state = 'asdiyvls'
    for s in states.keys() :
        t = deepcopy(answer)
        t['text'] = t['text'].lower().replace(curr_state, s.lower())
        t['entities'] = [{'value' : j['value'].replace(curr_state, s.lower()),
                          'entity' : j['entity'],
                          'start' : j['start'],
                          'end' : j['end']}
                         for j in t['entities']]
        train['rasa_nlu_data']['common_examples'].append(t)
    if len(synonym) > 0 :
        train['rasa_nlu_data']['entity_synonyms'].append(synonym)
        print '*** New Synonym Added'
    with open('../Codes and More/question_training_data2.json', 'w') as json_data:
        json.dump(train, json_data)
        json_data.close()
    print '*** Re-Creating the Models'
    training_data = load_data('../Codes and More/question_training_data2.json')
    trainer = Trainer(RasaNLUConfig("../Codes and More/config_spacy.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('../Codes and More/', fixed_model_name = 'my_model')
    print '*** Building Interpreter'
    interpreter = Interpreter.load(model_directory, RasaNLUConfig("../Codes and More/config_spacy.json"))
    print '--- DONE ---'
    return interpreter

def get_intent_count(text) :
    try :
        intent_count = {'intent' : intent_synonym.keys(),
                        'count' : []}
        for i in intent_synonym.keys() :
            intent_synonym[i].append(i)
            intent_i = list(set(intent_synonym[i]))
            c = 0
            for j in intent_i :
                c += text.count(j)
            intent_count['count'].append(c)
        df = pd.DataFrame(intent_count)
        m = max(df['count'])
        if m <= 1.5 * numpy.mean(df['count']) :
            pred_intent = ''
        else :
            pred_intent = '--'.join(list(df[df['count'] >= 0.7 * m].reset_index()['intent']))
    except AttributeError :
        pred_intent = ''
    return pred_intent

def check_states(question) :
    check_1 = pd.DataFrame([k.lower() for k in word_tokenize(question)]).merge(state_df, how = 'inner', left_on = 0, right_on = 'state')
    if check_1.shape[0] > 0 :
        question = question.lower().replace(check_1.state[0], check_1.code[0])
        answer = check_answer(question)
        curr_state = check_1.code[0]
    else :
        answer = check_answer(question)
        entities = answer['entities']
        st_ind = [i for i in range(len(entities)) if  entities[i]['entity'] == 'state']
        if len(st_ind) > 0 :
            curr_state = answer['entities'][st_ind[0]]['value'].swapcase()
        else :
            curr_state = 'state_not_found'
    return {curr_state : question}

def pred_intent(question) :
    answer = check_answer(question)
    return answer['intent']['name']

def summarize1(df) :
    summ = dict.fromkeys(df.index)
    for k in df.index :
        a = df.sent[k]
        sentences = {}
        for (i,j) in enumerate(sent_tokenize(a)) :
            if len(j.split(' ')) > 7 :
                if j in sentences.keys() :
                    sentences[j] = min(i, sentences[j])
                else :
                    sentences[j] = i
            else :
                pass
        txt = str(' '.join(sorted(sentences, key = sentences.get)))
        outstring = gensim.summarization.summarize(text = txt, ratio = 0.2, split = False, word_count=0.25*len(word_tokenize(txt)))
        summ[k] = [df.Research_Title[k], outstring, txt]
    return summ

def summarize(df2_state, df2_state_1) :
    summ = dict.fromkeys(df2_state_1.index)
    for k in df2_state_1.index :
        d = df2_state[df2_state.Research_Title == df2_state_1.Research_Title[k]]
        d1 = {'sent' : [], 'sim' : [], 'n' : [], 'w' : []}
        for i in d.index :
            sent = sent_tokenize(d.sent[i])
            for j in range(len(sent)) :
                if len(word_tokenize(sent[j])) > 7 :
                    d1['sent'].append(sent[j])
                    d1['sim'].append((len(sent)-0) * d.sim[i]/len(sent))
                    d1['w'].append(len(word_tokenize(sent[j])))
                    d1['n'].append(1)
                else :
                    pass
        d2 = pd.DataFrame(d1)
        d3 = d2.groupby('sent').sum().reset_index().sort_values('sim', ascending = False)
        d3['new'] = d3.sim * (d3.n / d3.w)
        d4 = d3[(d3.new > d3.new.quantile(0.75)) | (d3.sim > d3.sim.quantile(0.65))].reset_index()
        small_text = '\n'.join(list(d4.sent))
        if len(small_text) > 500 :
            outstring = gensim.summarization.summarize(text = small_text, word_count=500, split = False)
            if outstring == '' :
                outstring = small_text
            text = '\n'.join(list(d3.sent))
        else :
            outstring = small_text
            text = '\n'.join(list(d3.sent))
        summ[k] = [df2_state_1.Research_Title[k], outstring, text]
    return summ

def get_answers(params) :
    question = params['question']
    answer = check_answer(params['question'])
    pred_state = params['state'].upper()
    pred_intent = params['intent']
    all_intents = [j['name'] for j in answer['intent_ranking']]
    if ('' in all_intents) :
        all_intents.remove('')
    elif (' ' in all_intents) :
        all_intents.remove(' ')
    else :
        pass
    global intent_synonym
    intent_synonym = {intent : [i for i in synonym.keys() if synonym[i] == intent] for intent in all_intents}
    intent_pred = []
    for t in range(len(reg.Parsed_LN_Content)) :
        if reg.Parsed_LN_Content[t] == '' :
            if reg.Parsed_LN_Content_New[t] == '' :
                text = reg.Parsed_Item[t]
                intent_pred.append(get_intent_count(text))
            else :
                text = reg.Parsed_LN_Content_New[t]
                intent_pred.append(get_intent_count(text))
        else :
            text = reg.Parsed_LN_Content[t]
            intent_pred.append(get_intent_count(text))
    global reg
    reg['Intent_pred'] = intent_pred
    pred_entity = answer['entities']
    print pred_intent
    #print pred_state.swapcase()
    k = {}
    reg_state = reg[(reg.Jurisdiction == pred_state) | (reg.Jurisdiction == 'United States')]
    if pred_state == 'DC' :
        reg_state = reg_state.reset_index()
    else :
        dc_states = ['District of Columbia' not in str(title) for title in reg_state.Research_Title]
        reg_state = reg_state[dc_states].reset_index()
    for i in range(len(reg_state.Intent_pred)) :
        try :
            if (pred_intent in reg_state.Intent_pred[i]) | (pred_intent in reg_state.Intent[i]) | (pred_intent in reg_state.Keyword[i]) :
                k[i] = (float(pred_intent in reg_state.Intent_pred[i]) / len(reg_state.Intent_pred[i].split('--')) * 1.0) + \
                (float(pred_intent in reg_state.Intent[i]) / len(reg_state.Intent[i].split('--')) * 1.0) + \
                (float(pred_intent in reg_state.Keyword[i]) / len(reg_state.Keyword[i].split('--')) * 1.0)
        except TypeError :
            pass
    filtered_pred = reg_state.iloc[k.keys()]
    filtered_pred['Count'] = k.values()
    filtered_pred = filtered_pred.sort_values('Count', ascending=False)
    filtered_pred1 = filtered_pred[filtered_pred.Jurisdiction == pred_state]
    filtered_pred1 = filtered_pred1[filtered_pred1.Count >= filtered_pred1.Count.mean()]
    filtered_pred2 = filtered_pred[filtered_pred.Jurisdiction == 'United States']
    filtered_pred2 = filtered_pred2[filtered_pred2.Count >= filtered_pred2.Count.mean()]
    filtered_pred = pd.concat([filtered_pred1, filtered_pred2], axis = 0).drop_duplicates().reset_index()
    inferred_vector1 = model.infer_vector(question, alpha = 0.25, min_alpha = 1e-5, steps = 100)
    inferred_vector2 = model.infer_vector(question.split(), alpha = 0.25, min_alpha = 1e-5, steps = 100)
    inferred_vector = 0.5 * (inferred_vector1 + inferred_vector2)
    sims = model.docvecs.most_similar([inferred_vector], topn = tag_df.shape[0])
    sims_df = pd.DataFrame(sims, columns=['tags', 'sim'])
    sims_df['tag'] = ['---' + i.split('---')[1] + '---' for i in sims_df.tags]
    if filtered_pred.shape[0] > 0 :
        filtered_pred['tag'] = ['---TAG*sYm#&%*' + str(i) + '---' for i in filtered_pred['index']]
        df1 = tag_df.merge(pd.DataFrame(filtered_pred[['tag', 'Jurisdiction', 'Research_Title']]), how='inner', on='tag')
        df2 = df1.merge(sims_df, how='inner', on=['tag', 'tags']).sort_values('sim', ascending = False).reset_index()
        if pred_state == 'United States' :
            df2_state = df2
            is_intent = [(pred_intent in str(df2_state.Keyword[t])) | (df2_state.sim[t] > 0.5) for t in df2_state.index.values]
            if (df2_state.shape[0] == 0) | (sum(is_intent) == 0) :
                outstring = '**** NO DOCUMENTATION FOR RELATED QUESTION AMONG FEDERAL REGULATIONS ***'
                detailed_outstring = ''
            else :
                outstring = 'no output yet'
        else :
            df2_state = df2[df2.Jurisdiction == str(pred_state.upper())]
            is_intent = [(pred_intent in str(df2_state.Keyword[t])) | (df2_state.sim[t] > 0.5) for t in df2_state.index.values]
            if (df2_state.shape[0] == 0) | (sum(is_intent) == 0) :
                outstring = '**** NO DOCUMENTATION FOR RELATED QUESTION IN MENTIONED STATE. LOOKING FOR FEDERAL REGULATIONS ***'
                detailed_outstring = ''
                df2_state = df2[df2.Jurisdiction == 'United States']
                is_intent = [(pred_intent in str(df2_state.Keyword[t])) | (df2_state.sim[t] > 0.5) for t in df2_state.index.values]
                if (df2_state.shape[0] == 0) | (sum(is_intent) == 0) :
                    outstring = '**** NO DOCUMENTATION FOR RELATED QUESTION IN MENTIONED STATE AS WELL AS AMONG FEDERAL REGULATIONS ***'
                    detailed_outstring = 'You may try for a different state'
                else :
                    pass
            else :
                outstring = 'no output yet'
    else :
        reg_state['tag'] = ['---TAG*sYm#&%*' + str(i) + '---' for i in reg_state['index']]
        df1 = tag_df.merge(pd.DataFrame(reg_state[['tag', 'Jurisdiction', 'Research_Title']]), how='inner', on='tag')
        df2 = df1.merge(sims_df, how='inner', on=['tag', 'tags']).sort_values('sim', ascending = False).reset_index()
        is_intent = [pred_intent in str(s) for s in df2.sent]
        df2 = df2[is_intent].reset_index()
        df2_state = df2[df2.Jurisdiction == pred_state]
        df2_us = df2[df2.Jurisdiction == 'United States']
        if df2.shape[0] == 0 :
            outstring = '**** NO DOCUMENTATION FOR RELATED QUESTION IN MENTIONED STATE AS WELL AS AMONG FEDERAL REGULATIONS ***'
            detailed_outstring = 'You may try for a different state'
        elif df2_state.shape[0] == 0 :
            outstring = '**** NO DOCUMENTATION FOR RELATED QUESTION IN MENTIONED STATE. LOOKING FOR FEDERAL REGULATIONS ***'
            detailed_outstring = ''
            df2_state = df2_us
        else :
            outstring = 'no output yet'
    df2_state = df2_state[pd.Series([isinstance(k, str) for k in df2_state.sent])]
    title = df2_state.groupby('Research_Title').mean().sort_values('sim', ascending = False).index[:3]
    df2_state_1 = {'Research_Title' : list(title)}
    s = []
    for i in df2_state_1['Research_Title'] :
        s.append(' '.join(df2_state[df2_state.Research_Title == i].sent[:7]))
    df2_state_1['sent'] = s
    df2_state_1 = pd.DataFrame(df2_state_1)
    out = summarize(df2_state, df2_state_1)
    if outstring == 'no output yet' :
        txt = '\n\n===================================================================\n'.join(['***' + i[0] + '***\n--------------------------------------------------------------------\n' + i[2] for i in out.values()])
        outstring = '\n\n===================================================================\n'.join(['***' + i[0] + '***\n--------------------------------------------------------------------\n' + i[1] for i in out.values()])
    elif outstring == '**** NO DOCUMENTATION FOR RELATED QUESTION IN MENTIONED STATE. LOOKING FOR FEDERAL REGULATIONS ***' :
        txt = '\n\n===================================================================\n'.join(['***' + i[0] + '***\n--------------------------------------------------------------------\n' + i[2] for i in out.values()])
        head = '**** NO DOCUMENTATION FOR RELATED QUESTION IN MENTIONED STATE. LOOKING FOR FEDERAL REGULATIONS ***\n\n'
        outstring = head + '\n\n===================================================================\n'.join(['\n\n***' + i[0] + '***\n--------------------------------------------------------------------\n' + i[1] for i in out.values()])
    else :
        txt = detailed_outstring
    return({'summarized_response' : outstring, 'detailed_response' : txt})
