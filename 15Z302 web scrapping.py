
# coding: utf-8

# In[1]:


import requests 
from bs4 import BeautifulSoup 
imageurl='http://psgim.ac.in/2017/01/full-time-faculty/'
images=requests.get(imageurl) 
res=BeautifulSoup(images.content,'html.parser') 
for i in list(res.select('div a')):    
    string=i.text;    
    if((len(string)!=0)and(i.text[0]=='D')and(i.text[1]=='R')):        
        print(i.text) 

