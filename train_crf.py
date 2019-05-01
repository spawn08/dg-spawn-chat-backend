import crf_entity
import spacy

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_md") 
    crf_entity.set_nlp(nlp)
    print(crf_entity.train('/opt/models/crf_en_data.json','spawn','en'))
