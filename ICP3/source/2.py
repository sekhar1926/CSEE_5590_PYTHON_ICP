from bs4 import BeautifulSoup
import requests
import re

page_link = 'https://en.wikipedia.org/wiki/Deep_learning'
page_response = requests.get(page_link, timeout=5)

page_content = BeautifulSoup(page_response.content, "html.parser")

textContent = []
f = open('file1.txt', 'w')
f.write( "title is" +page_content.title.string+ '\n')
for links in page_content.find_all('a',attrs={'href': re.compile("^http://")}):
    textContent.append(links.get('href'))
    f.write( repr(links.get('href')) + '\n')
print(textContent)



