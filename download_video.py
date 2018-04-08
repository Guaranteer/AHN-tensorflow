import json
import os

with open('./captions/activity_net.v1-3.min.json', 'r') as fr:
    data = json.load(fr)

database = data['database']

for key,item in database.items():
    url = item['url']
    print(url)
    os.system("proxychains youtube-dl "+url+" -o video/" + key+ '.webm')

