'''
어간 추출 및 표제어 추출
단어의 개수를 줄일 수 있는 기법(어간 추출, 표제어 추출)

형태소: 의미를 가진 가장 작은 단위
형태소에는 두 가지 종류가 있는데 이는 어간과 접사이다.
어간: 단어의 의미를 담는 단어의 핵심 부분
접사: 단어에 추가적인 의미를 주는 부분

표제어: 기본 사전형 단어 ex) am, are, is -> be
형태학적 파싱

어간                            표제어
Stemming                        Lemmatization
am → am                         am → be
the going → the go              the going → the going
having → hav                    having → have

한국어 용언(동사, 형용사): 어간 + 어미
'''

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# ◆◆◆◆◆◆◆◆◆◆ 표제어 추출 ◆◆◆◆◆◆◆◆◆◆
# n = WordNetLemmatizer()
# words = ['policy', 'doing', 'organization', 'have', 'going', 'love',
#          'lives', 'fly', 'dies', 'watched', 'has', 'starting']
# print([n.lemmatize(w) for w in words])

# print(n.lemmatize('dies', 'v'))
# print(n.lemmatize('watched', 'v'))
# print(n.lemmatize('has', 'v'))

# ◆◆◆◆◆◆◆◆◆◆ 어간 추출 ◆◆◆◆◆◆◆◆◆◆
# s = PorterStemmer()
# text = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
# words = word_tokenize(text)
# print(words)

# s = PorterStemmer()
# words = ['formalize', 'allowance', 'electricical']  # ['formal', 'allow', 'electric']
# print([s.stem(w) for w in words])