
# Downloading PSG IM page using requests

import requests
page = requests.get("http://psgim.ac.in/2017/01/full-time-faculty/")
page

# Using beautiful soup and extracting faculty names having PHD.

from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')
for i in list(soup.select("div a")):
    str=i.text
    if(len(str)==0):
        continue
    if(str[0]=='D' and str[1]=='R'):
        print(str)

