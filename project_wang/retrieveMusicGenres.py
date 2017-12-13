import sys
import requests
from sets import ImmutableSet

validGenres = ImmutableSet(['Pop', 'Alternative', 'Rock', 'Country', 'Heavy Metal', 'Hip Hop/Rap'])

API_KEY = '39480b57e7724eb67d0b456e796581f7'

def retrieveMusicGenre(trackID):
    url = "http://api.musixmatch.com/ws/1.1/track.get?track_id=" + trackID + "&apikey=" + API_KEY
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.get(url, headers=headers)
    trackGenre = 'None'
    if r.json()[u'message'][u'header'][u'status_code'] != 200:
        return trackGenre
    trackGenreList = r.json()[u'message'][u'body'][u'track'][u'primary_genres'][u'music_genre_list']
    for genre in trackGenreList:
        genreName = genre[u'music_genre'][u'music_genre_name']
        if genreName in validGenres:
            if trackGenre == 'None':
                trackGenre = genreName
            else:
                trackGenre = 'Mult'
    return trackGenre


sys.stdout = open('train_genres.txt', 'w')
f = open('mxm_dataset_train.txt','r')
while True:
    line = f.readline()
    line = line.rstrip()
    if not line: break
    #comment lines ignored
    if line[0] == '#': continue
    #% line refers to most common words/ we dont handle this in this script
    if line[0] == '%': continue
    # we have a normal line
    else:
        counter = 0
        for word in line.split(','):
            counter = counter + 1
            if counter == 2:
                trackGenre = retrieveMusicGenre(word)
                print trackGenre
            if counter >= 2:
                break

f.close()