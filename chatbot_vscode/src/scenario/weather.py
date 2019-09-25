# Author : Hyunwoong
# When : 2019-06-22
# Homepage : github.com/gusdnd852


import crawler.weather as crawler


def weather(named_entity):
    keyword_group = named_entity[0] # ['내일', '광명', '날씨', '알려줘']
    entity_group = named_entity[1]  # ['DATE', 'LOCATION', 'O', 'O']
    date = []
    location = []

    for k in zip(keyword_group, entity_group):  # zip: 키워드 그룹과 엔티티 그룹을 같이 나열
        if 'DATE' in k[1]:  # 날짜가 있으면 date에 저장
            date.append(k[0])
        elif 'LOCATION' in k[1]:    # 위치가 있으면 location에 저장
            location.append(k[0])

    if len(date) == 0:  # 날짜 얘기가 없으면 '오늘'로 하겠다.
        date.append('오늘')

    if len(location) == 0:  # 지역이 없으면 지역을 설정해주세요.
        while len(location) == 0:
            print('A.I : ' + '어떤 지역을 알려드릴까요?')
            print('User : ', end='', sep='')
            loc = input()
            if loc is not None and loc.replace(' ', '') != '':
                location.append(loc)

    if '오늘' in date:  # 카테고리: 오늘, 내일, 모레, 내일모레, 이번주 등
        return crawler.today_weather(' '.join(location))
    elif date[0] == '내일':
        return crawler.tomorrow_weather(' '.join(location))
    elif '모레' in date or '내일모레' in date:
        return crawler.after_tomorrow_weather(' '.join(location))
    elif '이번' in date and '주' in date:
        return crawler.this_week_weather(' '.join(location))
    else:
        return crawler.specific_weather(' '.join(location), ' '.join(date))
