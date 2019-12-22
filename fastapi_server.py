import spacy
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED
import uvicorn

import crf_entity
import train_model

app = FastAPI()
security = HTTPBasic()

cache = {}
nlp = None


async def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "onebotsolution" or credentials.password != "OneBotFinancialServices":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"})
    return True


async def load_models():
    global nlp
    models = {'1': 'spawn_en', '2': 'spawn_hi'}
    for loaded_model in models.values():
        train_model.load_keras_model(loaded_model)
    nlp = spacy.load("en_core_web_md")
    crf_entity.set_nlp(nlp)
    pass


@app.on_event("startup")
async def load():
    print("Loading model..")
    await load_models()


@app.get('/api/classify')
async def classify(q: str, model: str, lang: str,
                   dependencies=Depends(get_current_username)):
    sentence = q
    model_name = model
    if model_name is None:
        return ({'error': 'Incorrent parameter arguments', 'status': 'fail'})
    sentence = sentence.lower()
    if lang is None:
        lang = 'en'
    model_name = '{model_name}_{lang}'.format(model_name=model_name, lang=lang)
    if (sentence is not None):
        return_list = train_model.classifyKeras(sentence, model_name)
    else:
        return ({'message': 'query cannot be empty', 'status': 'error', 'model_name': model_name})
    return (return_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line utility for accepting port number')
    parser.add_argument('--port', type=int, help='Port number for running application')

    args = parser.parse_args()
    uvicorn.run(app, port=args.port)
