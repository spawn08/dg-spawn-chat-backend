import crf_entity
import spacy
import os

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_md")
    crf_entity.set_nlp(nlp)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_BASE_PATH = os.path.join(ROOT_DIR, 'opt/')
    print(crf_entity.train(MODEL_BASE_PATH + 'crf_hi_data.json', 'spawn', 'hi'))
