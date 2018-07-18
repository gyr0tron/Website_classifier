import requests
import urllib.parse as urlparse
from bs4 import BeautifulSoup
import re
import json

def process_domain(match):
 requested_domain = 'https://www.alexa.com/topsites/category/Top/'+match
 try:
     request_result = requests.get(requested_domain, timeout=10, stream=True)
     if request_result.status_code == 200 and request_result.text is not None:
         result_text = request_result.text
         soup = BeautifulSoup(result_text, "html.parser")
         table_content = [table.find('a') for table in soup.find_all('div', class_="td DescriptionCell")]
         title = [url.text for url in table_content]
         return (title)
 except Exception as e:
        print ("Exception: "+requested_domain+"^"+str(e))


def converting_url(initial_url):
    absolute_url=[]
    for url in initial_url:
        url = url if urlparse.urlparse(url).scheme != '' else 'http://www.' + url
        absolute_url.append(url.lower())
    return absolute_url


def reading_the_html_page(url_list):
    for url in url_list:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
                'Connection': 'keep-alive'}
            request_result = requests.get(url, timeout=30, stream=True, headers=headers)

            if request_result.status_code == 200 and request_result.text is not None:
                soup = BeautifulSoup(request_result.text, "html.parser")
                tag = soup.find('meta', attrs={'name': re.compile('keywords', re.I)})
                keywords = tag.get('content') if tag is not None and tag.get('content')is not None \
                                             and tag.get('content').strip() else None
                if keywords is None:
                    tag = soup.find('meta', attrs={'name': re.compile('description', re.I)})
                    description = tag.get('content') if tag is not None and tag.get('content')is not None \
                                             and tag.get('content').strip() else None
                    lookup.append(description) if description is not None else None
                else:
                    lookup.append(keywords) if keywords is not None else None
            else:
                print ('Page Not Open:\t'+url)
        except Exception as e:
            print ("URL:" + url + str(e))

fileObject = open('training.json', 'w+')
dict = {}
category = ["Adult"]

for match in category:
    lookup = []
    # get the top 50 sites url for the corresponding domain
    initial_url = process_domain(match)
    # convert the url into the absolute url
    absolute_url = converting_url(initial_url)
    # read the keywords from the each websites
    reading_the_html_page(absolute_url)
    # create the lookup for the each domain
    dict[match] = lookup
#     store the data into the json file
json.dump(dict, fileObject)

