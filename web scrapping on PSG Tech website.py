
# coding: utf-8

# In[1]:


import requests 
from bs4 import BeautifulSoup 


# In[2]:


link='http://psgim.ac.in/2017/01/full-time-faculty/'
p=requests.get(link) 
soup=BeautifulSoup(p.content,'html.parser') 
for itr in list(soup.select('div a')): 
     s=itr.text; 
     if((len(s)!=0)and(itr.text[0]=='D')and(itr.text[1]=='R')): 
         print(itr.text) 

