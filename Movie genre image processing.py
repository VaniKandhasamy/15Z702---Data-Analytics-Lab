
# coding: utf-8

# In[1]:


import csv 


# In[2]:


import requests


# In[26]:


with open('Z:\sem7\DA\MovieGenre.csv') as csvfile:
    csvrow=csv.reader(csvfile, delimiter=',')    
    for row in csvrow:         
            filename = row[2]       
            url = row[5]       
            count=0;       
            if(url=='Poster'):          
                continue;        
            print(url)         
            result = requests.get(url,stream=True)      
            if result.status_code == 200:         
                    image = result.raw.read()         
                    v1="Z:/sem7/DA/image1/"+filename+".jpg"       
                    open(v1,"wb").write(image)

