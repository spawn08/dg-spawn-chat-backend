import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
import requests
import crf_entity
from train_model import LoadModel
from train_model import classifyKeras, train_parallel

app = FastAPI()
security = HTTPBasic()

nlp = None
cache = {}
SEARCH_URL = 'https://api.cognitive.microsoft.com/bing/v7.0/search'

# es = Elasticsearch([{'host': 'localhost', 'port': 9200}], scheme='http')

async def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "onebotsolution" or credentials.password != "OneBotFinancialServices":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"})
    return True


'''def index_data(data):
    es.index('boltcargo', doc_type='wiki', body=data)'''

'''
async def load_models():
    global nlp
    models = {'1': 'spawn_en', '2': 'spawn_hi'}
    for loaded_model in models.values():
        train_model.load_keras_model(loaded_model)
    nlp = spacy.load("en_core_web_md")
    crf_entity.set_nlp(nlp)
    pass
'''


@app.on_event("startup")
async def load():
    global nlp
    print(tf.__version__)
    print("Loading model..")
    load_model = LoadModel()
    load_model.load_current_model()
    nlp = load_model.get_nlp()


@app.get('/api/classify')
async def classify(q: str, model: str, lang: str,
                   dependencies=Depends(get_current_username)
                   ):
    sentence = q
    model_name = model
    if model_name is None:
        return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
    sentence = sentence.lower()
    if lang is None:
        lang = 'en'
    model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
    if (sentence is not None):
        return_list = classifyKeras(sentence, model_name)
        # task.add_task(index_data, data=return_list)
    else:
        return ({'message': 'query cannot be empty', 'status': 'error', 'model_name': model_name})
    return return_list


@app.get('/api/train')
async def train(model_name: str, lang: str,
                dependencies=Depends(get_current_username)
                ):
    try:

        if model_name is None:
            return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if (model_name is None):
            return ({'message': 'Model name parameter is not defined / empty.',
                     'error': 'Model could not be trained',
                     'status': 'error'})
        if lang is None:
            lang = 'en'
        model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)

        train_msg = train_parallel(model_name)

    except Exception as e:
        print(e)
        return ({'message': 'Error processing request.',
                 'error': 'Model could not be trained', 'status': 'error',
                 'model_name': model_name})

    return train_msg

@app.get('/websearch')
async def entity_extract(q: str, count:str,
        dependencies=Depends(get_current_username)
        ):
    global cache
    if (cache.get(q) is not None):
        return (cache.get(q))

    results = requests.get(SEARCH_URL, params={'q':q, 'count':count},
            headers={'Ocp-Apim-Subscription-Key':'f5873c265b8247a7af3490e7648c6c37','BingAPIs-Market':'en-IN'})
    results = results.json()
    cache[q] = results
    return results

@app.get('/entity_extract')
async def entity_extract(q: str, model: str, lang: str,
                         dependencies=Depends(get_current_username)
                         ):
    try:
        global cache
        nlp = crf_entity.get_nlp()
        entities = []
        labels = {}

        #if (cache.get(q) is not None):
        #    return (cache.get(q))

        if model is None:
            return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
        if lang is None:
            lang = 'en'

        model_name = '{model_name}_{lang}'.format(model_name=model, lang=lang)
        print(model_name)
        ml_response = classifyKeras(q, model_name)
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line utility for accepting port number')
    parser.add_argument('--port', type=int, help='Port number for running application')
    parser.add_argument('--host', type=str, help='Hostname for the application')
    args = parser.parse_args()
    uvicorn.run(app,port=args.port,host=args.host)

'''
def test():
    time.sleep(10)
    print('done')


@app.get("/test")
async def test_back(task: BackgroundTasks):
    task.add_task(test)
    return "ok"
'''
