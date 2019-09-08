# -*- coding: utf-8 -*-

from gensim.models import word2vec
import logging
import sys

# 로그 저장 용도
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# 첫번째 파라미터는 읽어올 파일의 이름
sentences = word2vec.LineSentence(sys.argv[1])
# size: 공간 크기, min_count: 단어 최저 등장 횟수, window: 윈도우 수
model = word2vec.Word2Vec(sentences,
                          size=100,
                          min_count=1,
                          window=10)
# 두번째 파라미터는 생성할 모델명
model.save(sys.argv[2])