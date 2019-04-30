import json
import os
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
import tensorflow as tf

import keras
from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from multiprocessing.pool import ThreadPool
#import spacy

pool = ThreadPool(processes=20)

#nlp = spacy.load("en_core_web_md")

stemmer = PorterStemmer()
words = {}
classes = {}
train_x_dict = {}
train_y_dict = {}
multiple_models = {}
graph = tf.get_default_graph()

def load_keras_model(model_name):
    global words
    global classes
    global documents
    global train_x
    global train_y
    global multiple_models

    multiple_models[model_name] = None
    model_path = "/opt/models/{model_name}/{model_name}.json".format(
        model_name=model_name)
    model_path_h5 = '/opt/models/{model_dir}/{model_name}.h5'.format(
        model_dir=model_name,
        model_name=model_name)
    if (os.path.isfile("/opt/models/{model_name}/{model_name}.json".format(
            model_name=model_name))):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
  
    my_file = Path("/opt/models/{model_name}/{name}".format(
        model_name=model_name, name=model_name))
    if my_file.is_file():
        data = pickle.load(open("/opt/models/{model_name}/{name}".format(
            model_name=model_name, name=model_name), "rb"))
        words[model_name] = data['words_{model}'.format(model=model_name)]
        classes[model_name] = data['classes_{model}'.format(model=model_name)]
        train_x_dict[model_name] = data['train_x_{model}'.format(model=model_name)]
        train_y_dict[model_name] = data['train_y_{model}'.format(model=model_name)]
        print("Loaded model from disk")


pass


def get_model_keras(model_name):
    train_x = train_x_dict[model_name]
    train_y = train_y_dict[model_name]
    model_nn = Sequential()
    model_nn.add(Dense(12, input_dim=len(train_x[0]), activation='relu'))
    model_nn.add(Dense(8, activation='relu'))
    model_nn.add(Dense(len(train_y[0]), activation='softmax'))
    return model_nn


def train_keras(model_name):
    global graph
    global ignore_words
    output_data = []
    words_list = []
    inputclasses = []
    documents_vocab = []
    train_xinput = []
    train_youtput = []
    tf.reset_default_graph()
    #f = open("/opt/training_data/data_{model_name}.csv".format(
    #    model_name=model_name), 'rU')

    #for line in f:
    #    cells = line.split(",")
    #    output_data.append((cells[0], cells[1]))

    #f.close()

    with open('/opt/data/{model}_data.json'.format(model=model_name)) as f:
        data = json.load(f)
        
    output_data = list((data.get('rasa_nlu_data').get('common_examples')))
    print(output_data)
    print("%s sentences in training data" % len(output_data))

    ignore_words = ['?']
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
    training = np.array(training)
    train_xinput = list(training[:, 0])
    train_youtput = list(training[:, 1])
    with graph.as_default():
        model_nn = Sequential()
        model_nn.add(Dense(12, input_dim=len(train_xinput[0]), activation='relu'))
        model_nn.add(Dense(8, activation='relu'))
        model_nn.add(Dense(len(train_youtput[0]), activation='softmax'))

        model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_nn.fit(np.array(train_xinput), np.array(train_youtput), epochs=120, batch_size=8)

        model_path = '/opt/models/{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)
        model_nn.save(model_path)

    pickle.dump(
        {'words_{model}'.format(model=model_name): words_list, 'classes_{model}'.format(model=model_name): inputclasses,
         'train_x_{model}'.format(model=model_name): train_xinput,
         'train_y_{model}'.format(model=model_name): train_youtput},
        open("/opt/models/{model_name}/{name}".format(
            model_name=model_name, name=model_name), "wb"))

    load_keras_model(model_name)
    return {'message': 'success', 'model_name': model_name}

def train_parallel(model_name):
    my_file = os.path.isdir(
        "/opt/models/{model_name}".format(model_name=model_name))
    if my_file == False:
        os.mkdir("/opt/models/{model_name}".format(model_name=model_name))
    async_train_result = pool.apply_async(train_keras, (model_name,))
    return async_train_result.get()


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


def classifyKeras(sentence, model_name):
    with graph.as_default():
        
        threshold = 0.70

        loaded_model = multiple_models.get(model_name)
        if loaded_model is None:
            multiple_models[model_name] = get_model_keras(model_name)
            loaded_model = multiple_models.get(model_name)

        file_path = '/opt/models/{model_dir}/{model_name}.h5'.format(
            model_dir=model_name,
            model_name=model_name)

        loaded_model.load_weights(file_path)

        result = loaded_model.predict(np.array([bow(sentence, words.get(model_name))]))[0]
        result = [[i, r] for i, r in enumerate(result) if r > 0.75]
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        return_intent = []
        intent = []
        return_probability = 0
        for r in result:
            return_list.append((classes.get(model_name)[r[0]], r[1]))
        print(return_list) 
        if (len(return_list) > 0):
            return_intent = return_list[0]
            intent = return_intent[0]
            return_probability = (return_intent[1])
            js = {
                "intent": {
                    "confidence": return_probability,
                    "name": intent
                },
                "text":sentence,
                "model":model_name
            }
        
            return (js)
        else:
            js = {
                "intent": {
                    "confidence": 0,
                    "name": None,
                },
                "text":sentence,
                "model":model_name
            }

        return (js)


def classify_parallel(sentence, model_name):
    async_train_result = pool.apply_async(classifyKeras, (sentence, model_name))
    return async_train_result.get()

#def get_ner():
#    entities = []
#    labels = {}
#    query = request.args.get('q')
#    if query is not None:
#        doc = nlp(query)
#        for ent in doc.ents:
#            labels['tag'] = ent.label_
#            labels['value'] = ent.text
#            labels['timestamp']= strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
#            entities.append(labels)
#            labels = {}
#            print(ent.text, ent.label_)
#        if (len(entities) == 0):
#            return ([{'tag': '', 'value': query}])
#    else:
#        return ([{'tag': '', 'value': query}])
#    return (entities)


