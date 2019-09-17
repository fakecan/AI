#
# from selenium import webdriver
# from time import sleep
#
# driver = webdriver.Firefox()
#
# driver.maximize_window() #브라우저 창 크기 최대화
#
# driver.get('https://login.11st.co.kr/auth/front/login.tmall?xfrom=&returnURL=https%3A%2F%2Fwww.11st.co.kr%2Fhtml%2Fmain.html')
#
# sleep(1)
#
# userid_ele = driver.find_element_by_id("loginName")
# userid_ele.clear()
# userid_ele.send_keys("aaaa")
#
#
# userpw_ele = driver.find_element_by_id("passWord")
# userpw_ele.clear()
# userpw_ele.send_keys("aaaa")
#
# sleep(3)
#
# loginbtn = driver.find_element_by_css_selector("form input[type='button']")
#
# loginbtn.click()
#
#
# driver.close()


from selenium import webdriver
from time import sleep

# driver = webdriver.Firefox()

path = "./Project/chromedriver.exe"

driver = webdriver.Chrome(path)

driver.maximize_window() #브라우저 창 크기 최대화

driver.get('https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com')

sleep(1)

userid_ele = driver.find_element_by_id("id")
userid_ele.clear()
userid_ele.send_keys("kanzibbul")


userpw_ele = driver.find_element_by_id("pw")
userpw_ele.clear()
userpw_ele.send_keys("")

sleep(5)

loginbtn = driver.find_element_by_css_selector("form input[type='submit']")

loginbtn.click()

driver.get('')#원하는 사이트로 이동

content = driver.find_element_by_tag_name("body")

html = content.text



driver.close()


