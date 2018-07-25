import re
from flask import jsonify
from flask import render_template
import nltk
import numpy as np
import tensorflow as tf
import tflearn
import json
import sys
import string
import unicodedata
import random
from nltk.stem.lancaster import LancasterStemmer
from bs4 import BeautifulSoup
import requests
from flask import Flask, redirect, url_for, request
app = Flask(__name__)
stemmer=LancasterStemmer()
@app.route('/success/<name>')
def success(name):


    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)).startswith('P'))


    #remove punctuations from sentences.
    def remove_punctuation(text):
        return text.translate(tbl)

    stemmer = LancasterStemmer()
    data = None

    # read the json file and load the training data
    with open('training.json') as json_data:
        data = json.load(json_data)
        # print(data)

    # get a list of all categories to train for
    categories = list(data.keys())
    words = []
    # a list of tuples with words in the sentence and category name
    docs = []

    for each_category in data.keys():
        for each_sentence in data[each_category]:

            each_sentence = remove_punctuation(each_sentence)
            # print(each_sentence)

            w = nltk.word_tokenize(each_sentence)
            # print("tokenized words: ", w)
            words.extend(w)
            docs.append((w, each_category))

    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))

    # print(words)
    # print(docs)

    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(categories)


    for doc in docs:
        bow = []
        token_words = doc[0]
        token_words = [stemmer.stem(word.lower()) for word in token_words]
        for w in words:
            bow.append(1) if w in token_words else bow.append(0)

        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1

        training.append([bow, output_row])

    random.shuffle(training)
    training = np.array(training)

    # trainX contains the Bag of words and train_y contains the label/ category
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    # model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    # model.save('model.tflearn')
    model.load('model.tflearn')


    # method to pre-process the keywords fetched from the url
    def get_tf_record(sentence):

        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        # bag of words
        bow = [0]*len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bow[i] = 1

        return(np.array(bow))
    s=get_tf_record(name)
    flat_list=[]
    probalities_value=model.predict([s])
    for sublist in probalities_value:
        for item in sublist:
            item=round(item,2)
            flat_list.append(item)

    # provide the data in the dictionary format to be rendered in the template
    d={}
    domains = ["Adult", "Shopping", "Business", "Science", "Computers", "Sports", "Health", "News"]
    i=0
    for keys in domains:

        d[keys]=flat_list[i]
        i=i+1
        
    print(d)
    return  render_template('results.html', result = d)


@app.route('/login',methods = ['POST'])
# method to fetch the keywords from the url
def login():
    try:
        if request.method == 'POST':
            url = request.form['nm']
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) '
                            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
                'Connection': 'keep-alive'}
            request_result = requests.get(url, timeout=30, stream=True, headers=headers)

            if request_result.status_code == 200 and request_result.text is not None:
                soup = BeautifulSoup(request_result.text, "html.parser")
                tag = soup.find('meta', attrs={'name': re.compile('keywords', re.I)})
                keywords = tag.get('content') if tag is not None and tag.get('content') is not None \
                                                and tag.get('content').strip() else None
                if keywords is None:
                    tag = soup.find('meta', attrs={'name': re.compile('description', re.I)})
                    description = tag.get('content') if tag is not None and tag.get('content') is not None \
                                                        and tag.get('content').strip() else None
                    lookup=(description) if description is not None else None
                else:
                    lookup=(keywords) if keywords is not None else None
            else:
                print ('Page Not Open:\t' + url)
    except Exception as e:
        print("URL:" + url + str(e))
    print('-----------------')
    print(lookup)
    print('-----------------')
    return redirect(url_for('success', name=lookup))






if __name__ == '__main__':
   app.run(debug = True)
