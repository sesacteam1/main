import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import time
import datetime

df = pd.DataFrame(columns=['회사명', '공고명', '채용 형태(경력, 신입)', '학력', '직장 위치', '키워드', '마감 기한', '공고 링크'])
page_no = 1
url = f"https://www.jobkorea.co.kr"
response = requests.get(url, headers={'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
print("Response Code:", response.status_code)

soup = bs(response.text, 'html.parser')

pages_element = soup.find('p', class_='filter-text')
if pages_element:
    pages = pages_element.find('strong').text.replace(',', '')
    pages = round(int(pages) / 20)
else:
    print("filter-text 요소를 찾을 수 없음")
    pages = 20  

for i in range(20):  
    url = f"https://www.jobkorea.co.kr"
    response = requests.get(url, headers={'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    soup = bs(response.text, 'html.parser')

    # HTML 요소 추출
    company_name = soup.find_all('a', class_='name dev_view') or []
    exp = soup.find_all('span', class_='exp') or []
    edu = soup.find_all('span', class_='edu') or []
    loc = soup.find_all('span', class_='loc long') or []
    date = soup.find_all('span', class_='date') or []
    etc = soup.find_all('p', class_='etc') or []
    info = soup.find_all('a', class_='title dev_view') or []
    for j in range(min(len(company_name), len(info), len(exp), len(edu), len(loc), len(date), len(etc))):
        df.loc[20 * i + j] = [
            company_name[j].text.strip() if j < len(company_name) else '',
            info[j].text.strip() if j < len(info) else '',
            exp[j].text.strip() if j < len(exp) else '',
            edu[j].text.strip() if j < len(edu) else '',
            loc[j].text.strip() if j < len(loc) else '',
            ','.join(etc[j].text.split(',')[:5]) if j < len(etc) else '',
            date[j].text.strip() if j < len(date) else '',
            "https://www.jobkorea.co.kr/" + info[j]['href'] if j < len(info) else ''
        ]

    page_no += 1
    time.sleep(5)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.jobkorea.co.kr/",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
}

url = "https://www.jobkorea.co.kr/Recruit/GI_Read/46555498"
response = requests.get(url, headers=headers)
print(soup.prettify())
