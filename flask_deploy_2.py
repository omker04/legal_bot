'''
flask server for sentiment analysis
'''

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
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import legal_alcohol_50_state_review


app = Flask(__name__)
cors = CORS(app)

@app.route('/heart_beat', methods = ['GET'])
@cross_origin(origin='*')
def heart_beat():
    return jsonify({'status':200})


@app.route('/alcohol', methods = ['POST'])
@cross_origin(origin='*')
def alcohol():
    question = request.json['query']
    intent = legal_alcohol_50_state_review.pred_intent(question)
    print intent
    if intent == 'affirm' :
        out = params['ext_summary']
        input_question = params['question']
        intent = params['intent']
        state = params['state']
        summ_answer = params['details']
        detail_answer = params['ext_summary']
        global params
        params = {'question' : '', 'state' : '', 'intent' : '', 'details' : '', 'ext_summary' : ''}
        return jsonify({'reply' : out, 'input_question' : input_question, 'intent' : intent, 'state' : state, 'summ_answer' : summ_answer, 'detail_answer' : detail_answer})
    elif intent == 'negative' :
        input_question = params['question']
        intent = params['intent']
        state = params['state']
        summ_answer = params['details']
        detail_answer = params['ext_summary']
        params = {'question' : '', 'state' : '', 'intent' : '', 'details' : '', 'ext_summary' : ''}
        global params
        return jsonify({'reply' : 'Thank You.', 'input_question' : input_question, 'intent' : intent, 'state' : state, 'summ_answer' : summ_answer, 'detail_answer' : detail_answer})
    elif intent == 'thanks' :
        return jsonify({'reply' : 'The Pleasure has been mine.'})
    elif intent == 'greet' :
        return jsonify({'reply' : 'Greetings for the day, dear User!'})
    elif intent == 'goodbye' :
        return jsonify({'reply' : 'Bye! Have a great day ahead.\nHope to chat with you soon.'})
    elif (params['question'] == '') :
        global parmas
        params['question'] = question
        state = legal_alcohol_50_state_review.check_states(params['question']).keys()[0]
        question = legal_alcohol_50_state_review.check_states(params['question']).values()[0]
        answer = legal_alcohol_50_state_review.check_answer(params['question'])
        if state == 'state_not_found' :
            return jsonify({'reply' : 'Please enter the 2-digit state code of the state that you would like to know the answers for. Enter "US" for Federal Regulations.'})
        else :
            global params
            params['state'] = state
    elif params['state'] == '' :
        if question.lower() == 'us' :
            global params
            params['state'] == 'United States'
        else :
            global params
            params['state'] = question.lower()
    elif params['intent'] == 'less_prob':
        intent = legal_alcohol_50_state_review.pred_intent(params['question'])
        answer = legal_alcohol_50_state_review.check_answer(params['question'])
        if question.lower() == 'ok' :
            global params
            params['intent'] = intent
            print params
            global interpreter
            legal_alcohol_50_state_review.interpreter = legal_alcohol_50_state_review.retrain_using_new_info(answer, 'OK')
        else :
            global params
            params['intent'] = question
            global interpreter
            legal_alcohol_50_state_review.interpreter = legal_alcohol_50_state_review.retrain_using_new_info(answer, question)
    else :
        pass
    print params['intent']
    answer = legal_alcohol_50_state_review.check_answer(params['question'])
    intent = legal_alcohol_50_state_review.pred_intent(params['question'])
    print params['intent']
    intent_confidence = answer['intent']['confidence']
    print intent_confidence
    if intent_confidence < 0.65 :
        global params
        params['intent'] = 'less_prob'
        out = 'Predicted question intent is %s. Confirm your intent with "OK" or enter the predicted intent in case of a wrong prediction.' % str(intent)
        return jsonify({'reply' : out,
                        'params' : params, 'intent' : intent, 'conf' : intent_confidence, 'state' : params['state'], 'summ_answer' : params['details'], 'detail_answer' : params['ext_summary']})
    else :
        global params
        params['intent'] = intent
        response = legal_alcohol_50_state_review.get_answers(params)
        global params
        params['ext_summary'] = response['detailed_response']
        params['details'] = response['summarized_response']
        return jsonify({'reply' : response['summarized_response'] + '\n\nWould you require more details on this? Please reply with "YES" or "NO".',
                        'input_question' : params['question'], 'intent' : params['intent'], 'state' : params['state'], 'summ_answer' : params['details'], 'detail_answer' : params['ext_summary']})

@app.route('/alcohol/feedback/insert', methods = ['POST'])
@cross_origin(origin='*')
def insert() :
    question = [request.json['input_question'].encode('utf-8').strip()]
    state = [request.json['state'].encode('utf-8').strip()]
    intent = [request.json['intent'].encode('utf-8').strip()]
    summ_answer = [request.json['summ_answer'].encode('utf-8').strip()]
    detail_answer = [request.json['detail_answer'].encode('utf-8').strip()]
    thumbs_up = [str(request.json['thumbs_up'])]
    new_row = {'question':question, 'state':state, 'intent':intent, 'summ_answer':summ_answer, 'detail_answer':detail_answer, 'thumbs_up':thumbs_up}
    df = pd.read_csv('./feedback.csv')
    df = pd.concat([df, pd.DataFrame(new_row)], axis = 0)
    df.to_csv('./feedback.csv', index = False)
    print 'result updated'
    return thumbs_up[0]


if __name__ == '__main__':
    global params
    params = {'question' : '', 'state' : '', 'intent' : '', 'details' : '', 'ext_summary' : ''}
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5001)
