#import csv
#import requests
#from bs4 import BeautifulSoup

#url = 'http://www.psgtech.edu/facgen.php?id=C503'
#for i in range(0,1):
 #   response = requests.get(url)
 #   html = response.content

  #  soup = BeautifulSoup(html)
   ## table = soup.find('table', attrs={'class': 'cv-section-title'})
   # print(soup)
    ##list_of_rows = []
    ##for row in table.find_all('h1')[1:]:
    ##    list_of_cells = []
    ##    for cell in row.find_all('h5'):
    ##        text = cell.text.replace('&nbsp;', '')
    ##        list_of_cells.append(text)
    ##    list_of_rows.append(list_of_cells)

    ##outfile = open("./data.csv", "wb")
    ##writer = csv.writer(outfile)
 
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