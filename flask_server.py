from time import gmtime, strftime

import requests
from elasticsearch import Elasticsearch
from flask import Flask, request, json, Response, jsonify

import crf_entity
import train_model

# es = Elasticsearch([{'host': 'api.spawnai.com'}], scheme='https')
es = Elasticsearch([{'host': 'localhost', 'port': 9200}], scheme='http')

app = Flask(__name__)
cache = {}


# nlp = None

def check_auth(username, password):
    return username == 'onebotsolution' and password == 'OneBotFinancialServices'


def authenticate():
    message = {'message': 'You are not authorized user to access this url'}
    return Response(json.dumps(message), mimetype='application/json')


@app.before_request
def check_authorization():
    header = request.authorization
    content_type = request.content_type
    if content_type != 'application/json':
        return jsonify({'message': '415 Unsupported Media Type'})
    if not header:
        return authenticate()
    elif not check_auth(header.username, header.password):
        return authenticate()


'''
def load_models():
    global nlp
    models = {'1': 'spawn_en', '2': 'spawn_hi'}
    for loaded_model in models.values():
        train_model.load_keras_model(loaded_model)
    nlp = spacy.load("en_core_web_md")
    crf_entity.set_nlp(nlp)
    pass
'''


@app.route('/api/getFile', methods=["GET"])
def get_file():
    file_name = request.args.get('fileName')

    if (file_name is not None):
        res = es.get('spawnai', doc_type='spawnai_file', id=file_name)
        bot_data_response = res['_source']
        print(bot_data_response)
        return jsonify(bot_data_response)
    else:
        print('Error')
        return jsonify({'msg': 'Error processing request', 'status': 'false'})


@app.route("/get_ner", methods=['GET'])
def get_ner():
    query = request.args.get('q')
    if query is not None:
        response1 = requests.get("https://spawnai.com/entity?q={query}".format(query=query),
                                 headers={"Authorization": "Basic c3Bhd25haTpzcGF3bjE5OTI="})
        resp = response1.json()
        wiki_response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{name}".format(name=(resp[0]).get('value')))
        print(resp)
        es.index('spawnai', doc_type='wiki', id=(resp[0]).get('value'), body=wiki_response.json())
        return jsonify(wiki_response.json())
    else:
        return jsonify({'msg': 'query cannot be empty', 'status': 'false'})
    return jsonify({"answer": "42"})


@app.route("/post_wiki", methods=['POST'])
def post_data():
    body = request.json
    print(body)
    body['timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    title = body.get('title')
    intent = body.get('intent')
    print(title)
    if body is not None and title is not None:
        es.index('spawnai', doc_type='doc', id=body.get('title'), body=body)
        es.index('spawnai', doc_type='wiki', body=body)
        return jsonify({'msg': 'success', 'status': 'true'})
    elif intent is not None:
        es.index('spawnai', doc_type='intent', body=body)
    else:
        return jsonify({'msg': 'query cannot be empty', 'status': 'false'})
    return jsonify({'msg': 'Error processing request', 'status': 'false'})


@app.route('/api/train', methods=['GET'])
def train():
    try:
        model_name = request.args.get('model_name')
        lang = request.args.get('lang')
        if model_name is None:
            return jsonify({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if (model_name is None):
            return jsonify(
                {'message': 'Model name parameter is not defined / empty.', 'error': 'Model could not be trained',
                 'status': 'error'})
        if lang is None:
            lang = 'en'
        model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)

        train_msg = train_model.train_parallel(model_name)

    except Exception as e:
        print(e)
        return jsonify(
            {'message': 'Error processing request.', 'error': 'Model could not be trained', 'status': 'error',
             'model_name': model_name})

    return jsonify(train_msg)


@app.route('/api/classify', methods=["GET"])
def classify():
    sentence = request.args.get('q')
    model_name = request.args.get('model')
    lang = request.args.get('lang')
    if model_name is None:
        return jsonify({'error': 'Incorrent parameter arguments', 'status': 'fail'})
    sentence = sentence.lower()
    if lang is None:
        lang = 'en'
    model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
    if (sentence is not None):
        return_list = train_model.classifyKeras(sentence, model_name)
    else:
        return jsonify({'message': 'query cannot be empty', 'status': 'error', 'model_name': model_name})
    return jsonify(return_list)


@app.route('/entity_extract', methods=['GET'])
def get_ner_test():
    try:
        global cache
        nlp = crf_entity.get_nlp()
        entities = []
        labels = {}
        query = request.args.get('q')
        if (cache.get(query) is not None):
            return jsonify(cache.get(query))
        model_name = request.args.get('model')
        lang = request.args.get('lang')
        if model_name is None:
            return jsonify({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if lang is None:
            lang = 'en'

        model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
        print(model_name)
        ml_response = train_model.classifyKeras(query, model_name)
        print(ml_response)
        if query is not None:
            if lang == 'en':
                doc = nlp(query)
                if len(doc.ents):
                    ent = doc.ents[0]
                    labels['tag'] = ent.label_
                    labels['entity'] = ent.text
                    entities.append(labels)
                    labels = {}
                    print(ent.text, ent.label_)

                    ml_response['entities'] = entities
                    cache[query] = ml_response
                else:
                    crf_ent = crf_entity.predict(query, lang, model_name)
                    print(crf_ent)
                    if (crf_ent.get('entities') is not None and len(list(crf_ent.get('entities').keys())) > 0 and len(
                            list(crf_ent.get('entities').values())[0]) > 0):
                        entities = [{'tag': list(crf_ent.get('entities').keys())[0],
                                     'value': list(crf_ent.get('entities').values())[0]}]
                    ml_response['entities'] = entities
                    cache[query] = ml_response
                    print(ml_response)
                    return jsonify(ml_response)
            else:
                crf_ent = crf_entity.predict(query, lang, model_name)
                print(crf_ent)
                if (crf_ent.get('entities') is not None and len(list(crf_ent.get('entities').keys())) > 0 and len(
                        list(crf_ent.get('entities').values())[0]) > 0):
                    entities = [{'tag': list(crf_ent.get('entities').keys())[0],
                                 'value': list(crf_ent.get('entities').values())[0]}]
                ml_response['entities'] = entities
                cache[query] = ml_response
                print(ml_response)
                return jsonify(ml_response)
        else:
            entities = [{'tag': '', 'value': ''}]
            ml_response['entities'] = entities
            cache[query] = ml_response
            return jsonify(ml_response)

    except Exception as e:
        print(e)
        return jsonify({'error': 'No model found. Please train the model first.', 'status': 'fail'})

    return jsonify(ml_response)


'''
# Celery Task Code
from celery import Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
@celery.task
def my_background_task():
    # some long running task here
    time.sleep(1)
    return {'result': 'ok'}
from celery import Celery

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])


@celery.task
def my_background_task():
    # some long running task here
    # time.sleep(1)
    es.index('boltcargo', doc_type='wiki', body={'name': 'hey spawn', 'status': 'success'})
    return {'result': 'ok'} '''
