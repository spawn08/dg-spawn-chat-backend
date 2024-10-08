import json
import os
import pickle
import random
import asyncio
import aiofiles

import nltk
import numpy as np
import spacy
import tensorflow as tf
from pathlib import Path
from elasticsearch import AsyncElasticsearch
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from nltk.stem.porter import PorterStemmer
from service import crf_entity, utils

ROOT_DIR = ''
es = AsyncElasticsearch(['localhost'], scheme='http', port=9200, http_auth=('spawnai_elastic', 'Spawn@#543'))

MODEL_BASE_PATH = ''
DATA_BASE_PATH = ''
# nlp = spacy.load("en_core_web_md")

graph = tf.compat.v1.get_default_graph()
stemmer = PorterStemmer()
words = {}
classes = {}
train_x_dict = {}
train_y_dict = {}
multiple_models = {}


class LoadModel():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        crf_entity.set_nlp(self.nlp)
        self.models = {}

    def load_current_model(self):
        self.models = {'1': 'spawn_en', '2': 'spawn_hi'}
        for loaded_model in self.models.values():
            load_keras_model(loaded_model)
        print("Success")

    def get_nlp(self):
        return self.nlp

def set_root_dir(root_dir):
    global ROOT_DIR
    global MODEL_BASE_PATH
    global DATA_BASE_PATH

    ROOT_DIR = root_dir
    MODEL_BASE_PATH = os.path.join(ROOT_DIR, 'opt/models/')  # '/opt/models/'
    DATA_BASE_PATH = os.path.join(ROOT_DIR, 'opt/data/')  # '/opt/data/'
    pass

def load_keras_model(model_name: str):
    global words
    global classes
    global documents
    global train_x
    global train_y
    global multiple_models

    multiple_models[model_name] = None

    my_file = Path(MODEL_BASE_PATH + "{model_name}/{name}".format(
        model_name=model_name, name=model_name))
    if my_file.is_file():
        data = pickle.load(open(MODEL_BASE_PATH + "{model_name}/{name}".format(
            model_name=model_name, name=model_name), "rb"))
        words[model_name] = data['words_{model}'.format(model=model_name)]
        classes[model_name] = data['classes_{model}'.format(model=model_name)]
        train_x_dict[model_name] = data['train_x_{model}'.format(model=model_name)]
        train_y_dict[model_name] = data['train_y_{model}'.format(model=model_name)]
        print("Loaded model from disk")
pass


def get_model_keras(model_name: str, file_path: str):
    train_x = train_x_dict[model_name]
    train_y = train_y_dict[model_name]
    model_nn = Sequential()
    model_nn.add(Dense(32, input_dim=len(train_x[0]), activation='relu'))
    model_nn.add(Dense(16, activation='relu'))
    model_nn.add(Dense(len(train_y[0]), activation='softmax'))
    model_nn.load_weights(file_path)
    return model_nn


async def train_keras(model_name: str, training_data: dict, training_type: str, reg_id: str, username: str):
    global ignore_words
    output_data = []
    words_list = []
    inputclasses = []
    documents_vocab = []
    train_xinput = []
    train_youtput = []

    my_file = os.path.isdir(
        MODEL_BASE_PATH + "{model_name}".format(model_name=model_name))
    if my_file == False:
        os.mkdir(MODEL_BASE_PATH + "{model_name}".format(model_name=model_name))
        
    if training_type == 'elastic':
        data = es.get('spawnai_file', doc_type='file', id=training_data)
        data = data['_source']
        print("Loaded ES Data")
    else:
        async with aiofiles.open(DATA_BASE_PATH + f'/{model_name}_data.json', 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

    output_data = list(data.get('intents'))
    print(output_data)
    print("%s sentences in training data" % len(output_data))

    ignore_words = ['?']
    # ignore_words = ['?',',','!','do','who','what','is','a','an','the','how','he']
    print(ignore_words)
    for pattern in output_data:
        w = nltk.word_tokenize(pattern['text'])
        words_list.extend(w)
        documents_vocab.append((w, pattern['intent']))
        if pattern['intent'] not in inputclasses:
            inputclasses.append(pattern['intent'])

    words_list = [stemmer.stem(w.lower()) for w in words_list if w not in ignore_words]
    words_list = sorted(list(set(words_list)))

    inputclasses = sorted(list(set(inputclasses)))

    print(len(documents_vocab), "documents_vocab")
    print(len(inputclasses), "classes", inputclasses)
    print(len(words_list), "unique stemmed words", words_list)

    training = []
    output_data = []
    output_empty = [0] * len(inputclasses)

    for doc in documents_vocab:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words_list:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[inputclasses.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype="object")
    train_xinput = list(training[:, 0])
    train_youtput = list(training[:, 1])
    model_nn = Sequential()
    model_nn.add(Dense(32, input_dim=len(train_xinput[0]), activation='relu'))
    model_nn.add(Dense(16, activation='relu'))
    model_nn.add(Dense(len(train_youtput[0]), activation='softmax'))

    await asyncio.to_thread(model_nn.compile, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    await asyncio.to_thread(model_nn.fit, np.array(train_xinput), np.array(train_youtput), epochs=95, batch_size=8)

    model_path = MODEL_BASE_PATH + '{model_dir}/{model_name}.h5'.format(
        model_dir=model_name,
        model_name=model_name)
    await asyncio.to_thread(model_nn.save, model_path)

    metadata_path = MODEL_BASE_PATH + "{model_name}/{name}".format(model_name=model_name, name=model_name)
    async with aiofiles.open(metadata_path, 'wb') as f:
        await asyncio.to_thread(
            pickle.dump,
            {'words_{model}'.format(model=model_name): words_list, 
             'classes_{model}'.format(model=model_name): inputclasses,
             'train_x_{model}'.format(model=model_name): train_xinput,
             'train_y_{model}'.format(model=model_name): train_youtput}, f)
    await asyncio.to_thread(load_keras_model, model_name)
    await utils.send_notification(reg_id, username)
    return {'message': 'success', 'model_name': model_name}


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (bag)


async def predict(sentence, model_name):

    with graph.as_default():
        file_path = MODEL_BASE_PATH + '{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)

        loaded_model = multiple_models.get(model_name)
        if loaded_model is None:
            load_keras_model(model_name)
            multiple_models[model_name] = get_model_keras(model_name, file_path)
            loaded_model = multiple_models.get(model_name)

        result = loaded_model.predict(
            np.array([bow(sentence, words.get(model_name))]))[0]
        class_integer = np.argmax(result)
        intent_class = classes.get(model_name)[class_integer]
        probability = result[class_integer]

        if probability > 0.70:

            js = {
                "intent": {
                    "confidence": str(probability),
                    "name": intent_class,
                },
                "text": sentence,
                "model": model_name
            }

            return js
        else:
            js = {
                "intent": {
                    "confidence": 0,
                    "name": None,
                },
                "text": sentence,
                "model": model_name
            }

        return js
