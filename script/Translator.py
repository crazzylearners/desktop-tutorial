import googletrans as gt
from translate import Translator
from textblob import TextBlob

def translate_1():
    lang = gt.LANGUAGES
    lang_names = list(lang.values())
    keys_ = list(lang.keys())
    return keys_,lang_names

def translate_2(input, to_convert):
    detect_lang = TextBlob(input).detect_language()
    translator = Translator(from_lang = detect_lang , to_lang = to_convert)
    result = translator.translate(input)
    return result