
# coding: utf-8

# In[1]:


import csv
import requests
with open('Z:\data analytics lab\MovieGenre.csv') as csvfile:
    csvrows=csv.reader(csvfile, delimiter=',')
    for row in csvrows:
        filename = row[2]
        url = row[5]
        count=0;
        if(url=='Poster'):
           continue;
        print(url)
        result = requests.get(url,stream=True)
        if result.status_code == 200:
            image = result.raw.read()
            v1="Z:/DA/image/"+filename+".jpg"
            open(v1,"wb").write(image);

