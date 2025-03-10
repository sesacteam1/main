import os
import time
import requests
import pytesseract
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ✅ 1. Selenium 설정
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # GUI 없이 실행 (백그라운드 실행)
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.178 Safari/537.36")

# # Mac에서 Chrome 실행 경로 설정 (필요한 경우)
# options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# WebDriver 실행
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# ✅ 2. 대상 웹페이지 열기
url = "https://www.jobkorea.co.kr/Search/?stext=%EA%B8%B0%ED%9A%8D%EA%B0%9C%EB%B0%9C"
driver.get(url)
time.sleep(5)  # 페이지 로딩 대기 (네트워크 상태에 따라 조절 가능)

# ✅ 3. <p> 태그 내부의 <img> 태그 모두 가져오기
img_elements = driver.find_elements(By.CSS_SELECTOR, "p img")
image_urls = [img.get_attribute("src") for img in img_elements]

# **디버깅: 찾은 이미지 URL 출력**
print("🔍 Found Image URLs:", image_urls)

# Selenium 종료
driver.quit()

# ✅ 4. 채용 공고 이미지 찾기 (로고 이미지 제외)
valid_image_urls = [url for url in image_urls if "jobkorea.co.kr/Mng/" in url]

if not valid_image_urls:
    print("❌ 채용 공고 이미지 찾기 실패 - 올바른 이미지를 찾지 못했습니다.")
    exit()

# ✅ 5. 첫 번째 유효한 이미지 다운로드
image_url = valid_image_urls[0]
image_path = "jobkorea_p_image.png"

# **User-Agent 헤더 추가**
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.178 Safari/537.36"
}

response = requests.get(image_url, headers=headers)
if response.status_code == 200:
    with open(image_path, "wb") as file:
        file.write(response.content)
    print(f"✅ 이미지 다운로드 완료: {image_path}")
else:
    print(f"❌ 이미지 다운로드 실패. 상태 코드: {response.status_code}")
    exit()

# ✅ 6. OCR 실행 (한글 + 영어 지원)
try:
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image, lang="eng+kor")
    print("📌 OCR 결과:\n", extracted_text)
except Exception as e:
    print(f"❌ OCR 실패: {e}")


-----------------------------------------------------------
The Kernel crashed while executing code in the current cell or a previous cell. 
Please review the code in the cell(s) to identify a possible cause of the failure. 
Click here for more info. 
View Jupyter log for further details.
-----------------------------------------------------------
          
