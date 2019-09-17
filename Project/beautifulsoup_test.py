from bs4 import BeautifulSoup
from urllib.request import urlopen

# html = urlopen("https://www.naver.com/")
# soup = BeautifulSoup(html, "html.parser")
# print(soup)

from selenium import webdriver
from time import sleep

path = "./Project/chromedriver.exe"
driver = webdriver.Chrome(path)

driver.get('https://weather.com/')