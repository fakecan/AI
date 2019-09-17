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
sleep(1)

user_ele = driver.find_element_by_class_name("theme__inputElement__4bZUj input__inputElement__1GjGE")
user_ele.clear()
user_ele.send_keys("Paris")

logbtn = driver.find_element_by_css_selector("form input[type=']")