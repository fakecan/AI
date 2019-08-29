#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 코드 1-1. Training Data 읽어오기

import pandas as pd
import numpy as np

trn = pd.read_csv('./data/train_ver2.csv')


# In[4]:


# 코드 1-2. Training Data 데이터 미리보기

trn.head()


# In[5]:


# 코드 1-3. 모든 변수 미리보기

for col in trn.columns:
    print('{}\n'.format(trn[col].head()))


# In[6]:


# 코드 1-4. Training Data 데이터 .info() 함수로 상세보기

trn.info()


# In[7]:


# 코드 1-5. 수치형 변수 살펴보기

num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64', 'float64']]
trn[num_cols].describe()


# In[8]:


# 코드 1-6. 범주형 변수 살펴보기

cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]
trn[cat_cols].describe()


# In[9]:


# 코드 1-7. 범주형 변수 고유값 출력해보기

for col in cat_cols:
    uniq = np.unique(trn[col].astype(str))
    print('-' * 50)
    print('# col {}, n_uniq {}, uniq {}'.format(col, len(uniq), uniq))


# In[1]:


# 시각화 준비
import matplotlib
import matplotlib.pyplot as plt
# Jupyter Notebook 내부에 그래프를 출력하도록 설정
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[11]:


# 코드 1-8. 변수를 막대 그래프로 시각화하기

skip_cols = ['ncodpers', 'renta']
for col in trn.columns:
    # 출력에 너무 시간이 많이 걸리는 두 변수는 skip
    if col in skip_cols:
        continue
        
    # 보기 편하게 영역 구분 및 변수명 출력
    print('='*50)
    print('col : ', col)
    
    # 그래프 크기 (figsize) 설정
    f, ax = plt.subplots(figsize=(20, 15))
    # seaborn을 사용한 막대그래프 생성
    sns.countplot(x=col, data=trn, alpha=0.5)
    # show() 함수를 통해 시각화
    plt.show()


# In[12]:


# 코드 1-9. 월별 금융 제품 보유 여부를 누적 막대 그래프로 시각화

# 날짜 데이터를 기준으로 분석하기 위하여, 날짜 데이터 별도 추출
months = np.unique(trn['fecha_dato']).tolist()
# 제품 변수 24개 추출
label_cols = trn.columns[24:].tolist()

label_over_time = []
for i in range(len(label_cols)):
    # 매월, 각 제품 변수의 총합을 label_sum에 저장
    label_sum = trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
    label_over_time.append(label_sum.tolist())
    
label_sum_over_time = []
for i in range(len(label_cols)):
    # 누적 막대 그래프 형식으로 시각화 하기 위하여, 누적값을 계산
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
    
# 시각화를 위한 색깔 지정
color_list = ['#F5B7B1','#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']

# 시각화를 위한 준비
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    # 24개 제품에 대하여 Histogram 그리기
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)

# 우측 상단에 Legend 추가하기
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], label_cols, loc=1, ncol = 2, prop={'size':16})


# In[13]:


# 코드 1-10. 누적 막대 그래프를 상대값으로 시각화하기

# label_sum_over_time의 값을 퍼센트 단위로 변환하기
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100

# 앞선 코드와 동일한, 시각화 실행 코드
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
    
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))],            label_cols, loc=1, ncol = 2, prop={'size':16})


# In[ ]:


# 제품 변수를 prods에 list형태로 저장한다
prods = trn.columns[24:].tolist()

# 날짜를 숫자로 변환하는 함수이다. 2015-01-28은 1, 2016-06-28은 18로 변환된다
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 날짜를 숫자로 변환하여 int_date에 저장한다
trn['int_date'] = trn['fecha_dato'].map(date_to_int).astype(np.int8)

# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가한다.
trn_lag = trn.copy()
trn_lag['int_date'] += 1
trn_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in trn.columns]

# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다. Lag 데이터의 int_date는 1 밀려있기 때문에, 저번달의 제품 정보가 삽입된다.
df_trn = trn.merge(trn_lag, on=['ncodpers','int_date'], how='left')

# 메모리 효율을 위해 불필요한 변수를 메모리에서 제거한다
del trn, trn_lag

# 저번달의 제품 정보가 존재하지 않을 경우를 대비하여 0으로 대체한다.
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)

# 원본 데이터에서의 제품 보유 여부 – lag데이터에서의 제품 보유 여부를 비교하여 신규 구매 변수 padd를 구한다
for prod in prods:
    padd = prod + '_add'
    prev = prod + '_prev'
    df_trn[padd] = ((df_trn[prod] == 1) & (df_trn[prev] == 0)).astype(np.int8)

# 신규 구매 변수만을 추출하여 labels에 저장한다.
add_cols = [prod + '_add' for prod in prods]
labels = df_trn[add_cols].copy()
labels.columns = prods
#labels.to_csv('../input/labels.csv', index=False)


# In[15]:


# 코드 1-12. 신규 구매 누적 막대 그래프를 시각화하기

#labels = pd.read_csv('../input/labels.csv').astype(int)
fecha_dato = trn['fecha_dato']

labels['date'] = fecha_dato.fecha_dato
months = np.unique(fecha_dato.fecha_dato).tolist()
label_cols = labels.columns.tolist()[:24]

label_over_time = []
for i in range(len(label_cols)):
    label_over_time.append(labels.groupby(['date'])[label_cols[i]].agg('sum').tolist())
    
label_sum_over_time = []
for i in range(len(label_cols)):
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
    
color_list = ['#F5B7B1','#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']

f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)
    
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], label_cols, loc=1, ncol = 2, prop={'size':16})


# In[16]:


# 코드 1-13. 신규 구매 누적 막대 그래프를 상대값으로 시각화하기

label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100

f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
    
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))],            label_cols, loc=1, ncol = 2, prop={'size':16})

