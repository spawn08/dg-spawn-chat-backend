import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

from train_model import LoadModel
from train_model import classifyKeras

app = FastAPI()
security = HTTPBasic()

nlp = None


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line utility for accepting port number')
    parser.add_argument('--port', type=int, help='Port number for running application')
    args = parser.parse_args()
    uvicorn.run(app)

'''
def test():
    time.sleep(10)
    print('done')


@app.get("/test")
async def test_back(task: BackgroundTasks):
    task.add_task(test)
    return "ok"
'''
