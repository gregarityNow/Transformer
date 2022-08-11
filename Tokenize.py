import spacy
import re

class tokenize(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


class CamOrLetterTokenizer():
    def __init__(self,camTok):
        self.camTok = camTok
    def cam_tokenize(self, sentence):
        return [str(x) for x in self.camTok.encode(sentence)]
    def letter_tokenize(self, sentence):
        return list(sentence)

