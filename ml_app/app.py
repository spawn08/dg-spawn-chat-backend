import json
import os
import tensorflow as tf
from aiohttp import ClientSession
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

from service import crf_entity
from service.utils import get_current_username, send_notification
from service.train_model import LoadModel
from service.train_model import classifyKeras, train_keras, set_root_dir
from config.config import Config as cfg

app = FastAPI()
security = HTTPBasic()

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
set_root_dir(ROOT_DIR)
nlp = None
cache = {}
web_cache = {}
news_cache = {}
entity_cache = {}
client_session = None

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}], scheme='http')

@app.on_event("startup")
async def load():
    global nlp
    print(tf.version)
    print("Loading model..")
    load_model = LoadModel()


@app.get('/')
async def index():
    return "Spawn AI v3.0"

@app.get('/api/classify', tags=['Bot Service'])
async def classify(q: str, model: str, lang: str,
                   dependencies=Depends(get_current_username)
                   ):
    '''Prediction API for Bot Service. 
    Here query `q` will be passed to the model to get back prediction results'''
    sentence = q
    model_name = model
    if model_name is None:
        return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
    sentence = sentence.lower()
    if lang is None:
        lang = 'en'
    model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
    if sentence is not None:
        return_list = await classifyKeras(sentence, model_name)
        # task.add_task(index_data, data=return_list)
    else:
        return ({'message': 'query cannot be empty', 'status': 'error', 'model_name': model_name})
    return return_list


@app.get('/api/train', tags=['Bot Service'])
async def train(model_name: str, lang: str,
                task: BackgroundTasks,
                reg_id: str = None, username: str = None,
                dependencies=Depends(get_current_username)
                ):
    try:

        if model_name is None:
            return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if model_name is None:
            return ({'message': 'Model name parameter is not defined / empty.',
                     'error': 'Model could not be trained',
                     'status': 'error'})
        if lang is None:
            lang = 'en'
        model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)

        task.add_task(train_keras, model_name, "", "")
    except Exception as e:
        print(e)
        return ({'message': 'Error processing request.',
                 'error': 'Model could not be trained', 'status': 'error',
                 'model_name': model_name})

    return {'message': 'success', 'model_name': model_name}


@app.get('/api/train_bot', tags=['Bot Service'])
async def train_bot(model_name: str, lang: str,
                task: BackgroundTasks,
                reg_id: str = None, username: str = None,
                train_type: str = "local",
                dependencies=Depends(get_current_username)
                ):
    try:

        if model_name is None:
            return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if model_name is None:
            return ({'message': 'Model name parameter is not defined / empty.',
                     'error': 'Model could not be trained',
                     'status': 'error'})
        if lang is None:
            lang = 'en'
        model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
        training_data = model_name + "_data"
        task.add_task(train_keras, model_name, training_data, train_type, reg_id, username)
    
    except Exception as e:
        print(e)
        return ({'message': 'Error processing request.',
                 'error': 'Model could not be trained', 'status': 'error',
                 'model_name': model_name})

    return {'message': 'success', 'model_name': model_name}

@app.get('/send_notification', tags=['Notification Service'])
async def notification(reg_id: str, task: BackgroundTasks):
    ''' Send Notification API will send notification to the mobile device identified by 
    `reg_id`. 
     '''
    try:
        task.add_task(send_notification, reg_id,"Notifcaiton")
        return "Success"
    except Exception as e:
        print(e)
        return "Notifcaiton Failure"        


@app.get('/websearch', tags=['Web Search API'])
async def websearch(q: str, count: str, result_type: str,
                    dependencies=Depends(get_current_username)
                    ):
    ''' This API will call the Azure backend service to get the web results for user query 'q'.
    Based on the type of query e.g. news results, web results identified by query parameter `result_type`,
    the response will be return to the client.
     '''
    global web_cache
    global news_cache
    global client_session
    if client_session == None:
        client_session = ClientSession()

    if result_type == 'search':
        params = {'q': q, 'count': count}
        headers = {'Ocp-Apim-Subscription-Key': 'f5873c265b8247a7af3490e7648c6c37', 'BingAPIs-Market': 'en-IN',
                   'User-Agent': 'Android'}

        if web_cache.get(q) is not None:
            return web_cache.get(q)
        else:
            async with client_session.get(cfg.SEARCH_URL, params=params, headers=headers) as resp:
                results = await resp.json()

            web_cache[q] = results
            return results
    elif result_type == 'news':
        if news_cache.get(q) is not None:
            return news_cache.get(q)
        else:
            params = {'q': q, 'count': count, 'mkt': 'en-IN'}
            headers = {'Ocp-Apim-Subscription-Key': 'f5873c265b8247a7af3490e7648c6c37', 'BingAPIs-Market': 'en-IN',
                       'User-Agent': 'Android'}

            async with client_session.get(cfg.NEWS_URL, params=params, headers=headers) as resp:
                results = await resp.json()

            news_cache[q] = results
            return results
    elif result_type == 'entity':
        if entity_cache.get(q) is not None:
            return entity_cache.get(q)
        else:
            async with client_session.get(cfg.ENTITY_URL, params={'q': q, 'mkt': 'en-IN'},
                                          headers={'Ocp-Apim-Subscription-Key': 'f5873c265b8247a7af3490e7648c6c37',
                                                   'User-Agent': 'Android'}) as resp:
                results = await resp.json()

            entity_cache[q] = results
            return results


@app.get('/clear_cache')
async def clear_cache(dependencies=Depends(get_current_username)):
    ''' Clear the cache for web and news results to avoid memory exhaustion.
    We need to clear the cache manually as of now. In future, we will have to automate this 
    process.
    '''
    global web_cache
    global news_cache
    web_cache.clear()
    news_cache.clear()
    return {'status': 'success'}


@app.get('/entity_extract', tags=['Bot Service'])
async def entity_extract(q: str, model: str, lang: str,
                         dependencies=Depends(get_current_username)
                         ):
    try:
        global cache
        nlp = crf_entity.get_nlp()
        entities = []
        labels = {}

        if cache.get(q) is not None:
            return (cache.get(q))

        if model is None:
            return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if lang is None:
            lang = 'en'

        model_name = '{model_name}_{lang}'.format(model_name=model, lang=lang)
        print(model_name)
        ml_response = await classifyKeras(q, model_name)
        print(ml_response)
        if q is not None:
            if lang == 'en':
                doc = nlp(q)
                if len(doc.ents):
                    ent = doc.ents[0]
                    labels['tag'] = ent.label_
                    labels['entity'] = ent.text
                    entities.append(labels)
                    labels = {}
                    print(ent.text, ent.label_)

                    ml_response['entities'] = entities
                    cache[q] = ml_response
                else:
                    crf_ent = crf_entity.predict(q, lang, model_name)
                    print(crf_ent)
                    if (crf_ent.get('entities') is not None and len(list(crf_ent.get('entities').keys())) > 0 and len(
                            list(crf_ent.get('entities').values())[0]) > 0):
                        entities = [{'tag': list(crf_ent.get('entities').keys())[0],
                                     'value': list(crf_ent.get('entities').values())[0]}]
                    ml_response['entities'] = entities
                    cache[q] = ml_response
                    print(ml_response)
                    return ml_response
            else:
                crf_ent = crf_entity.predict(q, lang, model_name)
                print(crf_ent)
                if (crf_ent.get('entities') is not None and len(list(crf_ent.get('entities').keys())) > 0 and len(
                        list(crf_ent.get('entities').values())[0]) > 0):
                    entities = [{'tag': list(crf_ent.get('entities').keys())[0],
                                 'value': list(crf_ent.get('entities').values())[0]}]
                ml_response['entities'] = entities
                cache[q] = ml_response
                print(ml_response)
                return ml_response
        else:
            entities = [{'tag': '', 'value': ''}]
            ml_response['entities'] = entities
            cache[q] = ml_response
            return ml_response
    except Exception as e:
        print(e)
        return ({'error': 'No model found. Please train the model first.', 'status': 'fail'})

    return ml_response
