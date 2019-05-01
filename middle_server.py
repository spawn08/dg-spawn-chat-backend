from sanic import Sanic
from sanic import response
from signal import signal, SIGINT
from sanic import request
import asyncio
import requests
from elasticsearch import Elasticsearch
from time import gmtime, strftime
import json
from functools import wraps
import base64
import train_model
import crf_entity
import spacy

es = Elasticsearch([{'host': 'api.spawnai.com'}],scheme='https')

import uvloop

app = Sanic(__name__)

models = {'1': 'spawn_en', '2': 'spawn_hi'}
loading_models = {}
cache = {}
for loaded_model in models.values():
    train_model.load_keras_model(loaded_model)
nlp = spacy.load("en_core_web_md")
crf_entity.set_nlp(nlp)


def check_authorization(request):
    url_pass = b"spawnai:spawn1992"
    encode_auth = base64.b64encode(url_pass)
    auth = b'Basic %s' % encode_auth
    if auth.decode('utf-8') == request.token:
        return True
    else:
        return False


def authorized():
    def decorator(f):
        @wraps(f)
        async def decorated_function(request, *args, **kwargs):
            isauthorized = check_authorization(request)
            if isauthorized:
                response1 = await f(request, *args, **kwargs)
                return response1
            else:
                message = {'message': 'You are not authorized user to access this url'}
                return response.text(str(message))

        return decorated_function

    return decorator


@app.route('/api/getFile', methods=["GET"])
@authorized()
async def classify(request):
    file_name = request.args.get('fileName')

    if (file_name is not None):
        res = es.get('spawnai', doc_type='spawnai_file', id=file_name)
        bot_data_response = res['_source']
        print(bot_data_response)
        return response.json(bot_data_response)
    else:
        print('Error')
        return response.json({'msg': 'Error processing request', 'status': 'false'})


@app.route("/get_ner", methods=['GET'])
@authorized()
async def test(request):
    query = request.args.get('q')
    resp = {}
    if query is not None:
        response1 = requests.get("https://spawnai.com/entity?q={query}".format(query=query),
                                 headers={"Authorization": "Basic c3Bhd25haTpzcGF3bjE5OTI="})
        resp = response1.json()
        wiki_response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{name}".format(name=(resp[0]).get('value')))
        print(resp)
        es.index('spawnai', doc_type='wiki', id=(resp[0]).get('value'), body=wiki_response.json())
        return response.json(wiki_response.json())
    else:
        return response.json({'msg': 'query cannot be empty', 'status': 'false'})
    return response.json({"answer": "42"})


@app.route("/post_wiki", methods=['POST'])
@authorized()
async def test(request):
    body = request.json
    print(body)
    body['timestamp'] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    title = body.get('title')
    intent = body.get('intent')
    print(title)
    resp = {}
    if body is not None and title is not None:
        es.index('spawnai', doc_type='doc', id=body.get('title'), body=body)
        es.index('spawnai', doc_type='wiki', body=body)
        return response.json({'msg': 'success', 'status': 'true'})
    elif intent is not None:
        es.index('spawnai', doc_type='intent', body=body)
    else:
        return response.json({'msg': 'query cannot be empty', 'status': 'false'})
    return response.json({'msg': 'Error processing request', 'status': 'false'})


@app.route('/api/train', methods=['GET'])
@authorized()
async def train(request):
    try:
        model_name = request.args.get('model_name')
        lang = request.args.get('lang')
        if (model_name is None):
            return (response.json(
                {'message': 'Model name parameter is not defined / empty.', 'error': 'Model could not be trained',
                 'status': 'error'}))
        if lang is None:
            lang = 'en'
        model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)

        train_msg = train_model.train_parallel(model_name)

    except Exception as e:
        print(e)
        return (response.json(
            {'message': 'Error processing request.', 'error': 'Model could not be trained', 'status': 'error',
             'model_name': model_name}))

    return response.json(train_msg)


@app.route('/api/classify', methods=["GET"])
@authorized()
async def classify(request):
    sentence = request.args.get('q')
    model_name = request.args.get('model')
    lang = request.args.get('lang')
    sentence = sentence.lower()
    if lang is None:
        lang = 'en'
    model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
    if (sentence is not None):
        return_list = train_model.classifyKeras(sentence, model_name)
    else:
        return response.json({'message': 'query cannot be empty', 'status': 'error', 'model_name': model_name})
    return response.json(return_list)


@app.route('/entity_extract', methods=['GET'])
@authorized()
async def get_ner_test(request):
    global cache
    entities = []
    labels = {}
    query = request.args.get('q')
    if (cache.get(query) is not None):
        return response.json(cache.get(query))
    model_name = request.args.get('model')
    lang = request.args.get('lang')
    if lang is None:
        lang = 'en'

    model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)

    ml_response = train_model.classifyKeras(query, model_name)

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
                return response.json(ml_response)
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
            return response.json(ml_response)
    else:
        entities = [{'tag': '', 'value': ''}]
        ml_response['entities'] = entities
        cache[query] = ml_response
        return response.json(ml_response)
    return response.json(ml_response)


#asyncio.set_event_loop(uvloop.new_event_loop())
#server = app.create_server(host="0.0.0.0", port=8010, return_asyncio_server=True)
#loop = asyncio.get_event_loop()
#task = asyncio.ensure_future(server)
#signal(SIGINT, lambda s, f: loop.stop())
#try:
#   loop.run_forever()
#except:
#   loop.stop()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010)
