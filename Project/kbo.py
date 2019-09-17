import requests
from bs4 import BeautifulSoup

#시작
req = requests.get('https://www.koreabaseball.com/TeamRank/TeamRank.aspx')

html = req.text

# print(html)
#끝

#구조화된 데이터를 메모리 상에 옮겨놓음
soup = BeautifulSoup(html, 'html.parser')

# print(soup)

trs = soup.select('#cphContents_cphContents_cphContents_udpRecord table tbody tr') #앞에 #은 id를 의미

# print(target)

for line in trs:
    # print(line)
    tds = line.select('td')
    print(tds[0].text, tds[1].text)
    print("--------------------------------------------------")