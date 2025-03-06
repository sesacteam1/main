import requests
from bs4 import BeautifulSoup
def crawl_jobkorea(url):
    """잡코리아 채용 공고 페이지에서 전체 텍스트를 크롤링합니다."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True) # 모든 텍스트 추출 및 줄바꿈으로 분리
        return text
    except requests.exceptions.RequestException as e:
        print(f"오류 발생: {e}")
        return None
# 크롤링할 URL
url = "https://www.jobkorea.co.kr/Search/?stext=%EA%B0%9C%EB%B0%9C%EA%B8%B0%ED%9A%8D"
# 크롤링 실행 및 결과 출력
crawled_text = crawl_jobkorea(url)
if crawled_text:
    print(crawled_text)
---------------------------------------------------------
회원님께서는 현재 입력할 수 없는 문자열의 사용으로 인해 차단이 되었습니다.
문제가 지속적으로 발생할 경우 아래 고객센터로 문의하시기 바랍니다.
이용에 불편을 드려 죄송합니다.
문의(고객센터): 1588-9350
----------------------------------------------------------
import requests
from bs4 import BeautifulSoup
url = "https://www.jobkorea.co.kr/Search/?stext=%EA%B0%9C%EB%B0%9C%EA%B8%B0%ED%9A%8D"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://asia.creativecdn.com/ig-membership?ntk=qGLSOtRPqcTS6EYPoT-rNdHMrsTU-eEl-1--98_hSU3TU7UBWNmkutkJ9yS2TuIhiKV6gKl4oARelT3DESqdN8LfouhSoVPebXotT712p_g", 
}
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    print(soup.get_text())
except requests.exceptions.RequestException as e:
    print(f"오류 발생: {e}")
----------------------------------------------------------
# 이걸로 돌리면 아웃풋은 확인 가능한데 다음 페이지 구현에서 애를 먹고 있음 
