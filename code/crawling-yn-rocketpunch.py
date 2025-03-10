from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import json

# 1. WebDriver 설정
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")  # 백그라운드 실행
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 2. RocketPunch 특정 페이지 접속
url = "https://www.rocketpunch.com/jobs/152510/%EC%95%8C%ED%8C%8C%EB%B8%8C%EB%9D%BC%EB%8D%94%EC%8A%A4-%EC%A0%9C%EC%9E%91-PD-%EA%B4%91%EA%B3%A0%ED%99%8D%EB%B3%B4-%EC%BD%98%ED%85%90%EC%B8%A0%EA%B8%B0%ED%9A%8D%EC%98%81%EC%83%81"
driver.get(url)

# 3. 페이지 로딩 대기
time.sleep(3)

# 4. `div.title` 가져오기
try:
    title_text = driver.find_element(By.CSS_SELECTOR, "div.title").text.strip()
except:
    title_text = "제목 없음"

# 5. `div.content` 여러 개 가져오기
try:
    content_elements = driver.find_elements(By.CSS_SELECTOR, "div.content")
    content_texts = [content.text.strip() for content in content_elements if content.text.strip()]
except:
    content_texts = ["내용 없음"]

# 6. 크롤링한 데이터 저장
data = {
    "title": title_text,
    "content": content_texts
}

# 7. JSON 파일로 저장
with open("rocketpunch_title_content.json", "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

# 8. WebDriver 종료
driver.quit()

# 9. 완료 메시지
print("JSON 파일 저장 완료: rocketpunch_title_content.json")
------------------------------------------------------------
마감, 지역, 경력, 산업분야 , 업무분야  , 연봉 , 근무지 , 복지혜택 수집 가능
