# 정제: 갖고 있는 코퍼스로부터 노이즈 데이터를 제거한다.
# 정규화: 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어준다.
import re
text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))
