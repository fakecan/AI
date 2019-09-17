from selenium import webdriver
from time import sleep

path = "./Project/chromedriver.exe"
driver = webdriver.Chrome(path)

# driver.get('https://www.google.com')
# userid_ele = driver.find_element_by_name("q")

driver.get('https://map.naver.com/')
sleep(2)

userid_ele = driver.find_element_by_id("search-input")
# userid_ele = driver.find_element_by_id("gsr")
# userid_ele = driver.find_element_by_name("queryInputCustom")

userid_ele.clear()
userid_ele.send_keys("파리 빵집")

# userid_ele.submit()

sleep(600)

driver.close()