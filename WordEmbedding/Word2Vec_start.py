import re
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

targetXML = open('./Data/ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>오ㅏ </content> 사이의 내용만 가져온다.

content_text = re.sub(r'\([^)]*\)', '', parse_text)
# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter)
# 등의 배경음 부분을 제거합니다.
# 해당 코드는 괄호로 구성된 내용을 제거하는 코드입니다.

sent_text = sent_tokenize(content_text)
# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행합니다.

normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", "", string.lower())
    normalized_text.append(tokens)
# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환합니다.

result = []
result = [word_tokenize(sentence) for sentence in normalized_text]
# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행합니다.

print(result[:10])
# 문장 10개만 출력

from gensim.models import Word2Vec
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)

a=model.wv.most_similar("man")
print(a)
