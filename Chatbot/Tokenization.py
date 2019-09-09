'''
main.py에서 application.py안의 run을 호출
run이 실행하면서 intent의 classifier.py안의 get_intent를 호출


@classifier.py
inference_embed에 입력한 텍스트가 들어가고 이를 numpy화


'''

'''
토큰화 고려 사항
1) 구두점이나 특수 문자를 단순히 제외해서는 안된다.
2) 줄임말이나 단어 내에 띄어쓰기가 있는 경우를 고려한다.
'''


import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, TreebankWordTokenizer, sent_tokenize
from nltk.tag import pos_tag
from konlpy.tag import Okt, Kkma
# ◆◆◆◆◆◆◆◆◆◆ 단어 토큰화 ◆◆◆◆◆◆◆◆◆◆
# nltk.download('book')   # data download
# print(word_tokenize(
# "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

# print(WordPunctTokenizer().tokenize(
# "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# WordPunctTokenizer는 구두점을 별도로 분류한다.

# tokenizer = TreebankWordTokenizer() # 표준
# text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own"
# print(tokenizer.tokenize(text))

# ◆◆◆◆◆◆◆◆◆◆ 문장 토큰화 ◆◆◆◆◆◆◆◆◆◆
# text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
# print(sent_tokenize(text))

# text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
# print(sent_tokenize(text))

# ◆◆◆◆◆◆◆◆◆◆ 영어 토큰화 ◆◆◆◆◆◆◆◆◆◆
# text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
# print(word_tokenize(text)) 
# x = word_tokenize(text)
# print(pos_tag(x))

# ◆◆◆◆◆◆◆◆◆◆ 한글 형태소 토큰화 ◆◆◆◆◆◆◆◆◆◆
# morphs: 형태소 추출, pos: 품사 태깅, nouns: 명사 추출

# okt = Okt()
# print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

# kkma = Kkma()
# print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
