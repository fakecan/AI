'''
불용어: 문장 내에서 별 도움이 되지 않는 단어 코튼을 제거하는 작업
'''
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ◆◆◆◆◆◆◆◆◆◆ 영어 불용어 제거 ◆◆◆◆◆◆◆◆◆◆
# print(stopwords.words('english')[:10])  # 실제 출력은 100개 이상

# example = "Family is not an important thing. It's everything."
# stop_words = set(stopwords.words('english'))

# word_tokens = word_tokenize(example)

# result = []
# for w in word_tokens:
#     if w not in stop_words:
#         result.append(w)

# print(word_tokens)
# print(result)

# ◆◆◆◆◆◆◆◆◆◆ 한글 불용어 제거 ◆◆◆◆◆◆◆◆◆◆
example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# stop_words = ['아무거나', '아무렇게나', '어찌하든지', '같다', '비슷하다', '예컨대', '이럴정도로', '하면', '아니거든']
# 이거 넣고 위아래 주석 처리해도 가능

stop_words = stop_words.split(' ')
word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens)
print(result)