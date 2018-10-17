
# coding: utf-8

# In[2]:


import requests
from bs4 import BeautifulSoup
 
def phdholders():   
    url='http://psgim.ac.in/2017/01/full-time-faculty/'
    resp=requests.get(url)
    if resp.status_code==200:
        soup=BeautifulSoup(resp.text,'html.parser')    
        l=soup.find("div",{"class":"wpb_wrapper"})
        count=0
        for i in l.findAll("a"):
            str=i.text
            if "Dr"  in str: 
                print(str)
                count=count+1
        print("The number of phd holders in psg im : ",count)        
    else:
        print("Error")
         
phdholders()

