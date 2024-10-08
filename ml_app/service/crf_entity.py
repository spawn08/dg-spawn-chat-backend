import json
import os

import joblib
import sklearn_crfsuite
from spacy.training import biluo_tags_to_offsets

nlp = None  # spacy.load("en_core_web_md")
crf = None
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE_PATH = os.path.join(ROOT_DIR, 'opt/models/')
DATA_BASE_PATH = os.path.join(ROOT_DIR, 'opt/data/')


def train(filePath, model_name, lang):
    try:
        global crf
        global nlp

        model_path = "{model_name}_{lang}_classifier".format(model_name=model_name, lang=lang)

        if not filePath.lower().endswith('json'):
            return {'success': False, 'message': 'Training file should be in json format'}
        with open(filePath, encoding="utf-8") as file:
            ent_data = json.load(file)
        dataset = [jsonToCrf(q, nlp) for q in ent_data['entity_examples']]
        X_train = [sent2features(s) for s in dataset]
        y_train = [sent2labels(s) for s in dataset]
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.01,
            c2=0.01,
            max_iterations=150,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)
        if (not os.path.exists(MODEL_BASE_PATH + 'crfModel/')):
            os.mkdir(MODEL_BASE_PATH + 'crfModel/')
        if (os.path.isfile(MODEL_BASE_PATH + 'crfModel/' + '{model}.pkl'.format(model=model_path))):
            os.remove(MODEL_BASE_PATH + 'crfModel/' + '{model}.pkl'.format(model=model_path))
        joblib.dump(crf, MODEL_BASE_PATH + 'crfModel/' + '{model}.pkl'.format(model=model_path))
        return {'success': True, 'message': 'Model Trained Successfully'}
    except Exception as ex:
        return {'success': False, 'message': 'Error while Training the model - ' + str(ex)}


def predict(utterance, lang, model_name):
    try:
        global crf
        global nlp
        crf_cache = {}
        tagged = []
        finallist = []

        model_path = "{model_name}_classifier.pkl".format(model_name=model_name)

        if len(utterance.split()) > 1:
            parsed = nlp(utterance)
            for i in range(len(parsed)):
                tagged.append((str(parsed[i]), parsed[i].tag_))
            finallist.append(tagged)
            test = [sent2features(s) for s in finallist]
            crf = crf_cache.get(model_path)
            if (os.path.isfile(
                            MODEL_BASE_PATH + 'crfModel/' + '{model_path}.pkl'.format(
                        model_path=model_path)) and crf is None):
                crf = joblib.load(
                    MODEL_BASE_PATH + 'crfModel/' + '{model_path}.pkl'.format(model_path=model_path))
                crf_cache[model_path] = crf
                print("CRF MODEL LOADED")
                predicted = crf.predict(test)
                entityList = extractEntities(predicted[0], tagged)
                return {'success': True, 'entities': entityList}
            elif crf is not None:
                predicted = crf.predict(test)
                print('Predicted entity --> {e}'.format(e=predicted))
                entityList = extractEntities(predicted[0], tagged)
                return {'success': True, 'entities': entityList}
            else:
                return {'success': False, 'entities': None}
        else:
            return {'success': False, 'entities': None}
    except Exception as ex:
        return {'success': False, 'entities': None}


def jsonToCrf(json_eg, spacy_nlp):
    global nlp
    entity_offsets = []
    doc = spacy_nlp(json_eg['text'])
    for i in json_eg['entities']:
        entity_offsets.append(tuple((i['start'], i['end'], i['entity'])))
    gold = biluo_tags_to_offsets(doc, entities=entity_offsets)
    ents = [l[5] for l in gold.orig_annot]
    crf_format = [(doc[i].text, doc[i].tag_, ents[i]) for i in range(len(doc))]
    return crf_format


def word2features(sent, i):
    global nlp
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def extractEntities(predicted, tagged):
    global nlp
    rslt = {}
    label = ''
    for i in range(len(predicted)):
        if predicted[i].startswith('U-'):
            label = tagged[i][0]
            try:
                rslt[predicted[i][2:]].append(label)
            except:
                rslt[predicted[i][2:]] = [label]
            label = ''
            continue
        if predicted[i].startswith('B-'):
            label += tagged[i][0] + " "
        if predicted[i].startswith('I-'):
            label += tagged[i][0] + " "
        if predicted[i].startswith('L-'):
            label += tagged[i][0]
            try:
                rslt[predicted[i][2:]].append(label)
            except:
                rslt[predicted[i][2:]] = [label]
            label = ''
            continue
    return rslt


def set_nlp(nlp_load):
    global nlp
    nlp = nlp_load


def get_nlp():
    return nlp
    # train("E:/dg-spawn-chat-backend/opt/crf_en_data.json","spawn","en")
