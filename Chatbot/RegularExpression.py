# https://wikidocs.net/21703
import re
import nltk
from nltk.tokenize import RegexpTokenizer

'''
# 1) . 기호
# r = re.compile("a.c")   # a와 c 사이의 어떤 1개의 문자라도 올 수 있다.
# print(r.search("abc"))

# 2) ? 기호
# r = re.compile("ab?c")
# print(r.search("abbc"))

# 3) * 기호
# 4) + 기호
# 5) ^ 기호
# 6) {숫자} 기호
# 7) {숫자1, 숫자2} 기호
# 8) {숫자,} 기호
# 9) [ ] 기호
# 10) [^문자] 기호
'''

# 1) re.match()

# 2) re.search()
# text = "사과 딸기 수박 메론 바나나"
# print(re.split(" ", text))

# 3) re.findall()
# text = "이름 : 김철수   전화번호 : 010 - 1234 - 1234    나이 : 30   성별 : 남"""
# print(re.findall("\d+", text))

# 4) re.sub()
# text = "Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
# print(re.sub('[^a-zA-Z]',' ',text))

# text = """100 John    PROF
# 101 James   STUD
# 102 Mac   STUD"""  
# print(re.split('\s+', text))


# tokenizer = RegexpTokenizer("[\w]+")    # [\w]: 단어들만 토큰화하고 구두점은 제외
# print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

# tokenizer = RegexpTokenizer("[\s]+", gaps=True)
# print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))