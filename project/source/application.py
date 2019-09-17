from util.hanspell.spell_checker import fix
from util.tokenizer import tokenize

from intent.classifier import get_intent



def run():
    while True:
        print('챗봇입니다. 어디로 여행을 가시겠어요? : ')
        print('User : ', sep='', end='')
        speech = preprocess(input())
        print('Preprocessed : ' + speech, sep='')
        intent = get_intent(speech)

def preprocess(speech) -> str:
    speech = fix(speech)
    speech = tokenize(speech)
    speech = fix(speech)
    return speech

def get_entity(intent, speech):




