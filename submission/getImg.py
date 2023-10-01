import requests
from bs4 import BeautifulSoup

def get_youtube_thumbnail(video_url):
    # 获取视频页面内容
    response = requests.get(video_url)
    content = response.text

    # 使用BeautifulSoup解析页面内容
    soup = BeautifulSoup(content, 'html.parser')

    # 查找视频封面标签
    thumbnail_tag = soup.find('meta', property='og:image')

    if thumbnail_tag:
        thumbnail_url = thumbnail_tag['content']
        return thumbnail_url

    return None

