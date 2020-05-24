import pyecharts
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
cookie = {"cityPy":"UM_distinctid=171f2280ef23fb-02a4939f3c1bd4-335e4e71-144000-171f2280ef3dab; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1588905651; CNZZDATA1275796416=871124600-1588903268-%7C1588990372; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1588994046"}
header = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3756.400 QQBrowser/10.5.4039.400"}
url = "http://lishi.tianqi.com/beijing/201801.html"
html = requests.get(url = url, headers = header, cookies=cookie)
soup = BeautifulSoup(html.content, 'html.parser')
a=soup.select('.linegraphbox ul div')
print(a)