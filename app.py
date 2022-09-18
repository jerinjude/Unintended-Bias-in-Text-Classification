from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
import nltk
import string
import spacy
spcy = spacy.load("en_core_web_sm")
import contractions
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
import emoji
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer
from textblob import TextBlob
import scipy.sparse 
from scipy.sparse import hstack



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
def preprocess_text(text):
    words=[contractions.fix(word) for word in text.split()]
    text=(' ').join(words)
    text=text.lower()
    text=re.sub(r'https:\S*','',text) #http
    text=re.sub(r'sh\*tty','shitty',text) 
    text=re.sub(r'(ha){2,}','laughing',text) #hahaha
    text=re.sub(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+',"",text) #email
    text=re.sub(r'\S+\.com','',text) #.com
    emoji.demojize(text, delimiters=("", ""))
    text=re.sub(r'\n',' ',text)
    text=re.sub(r'\\',' ',text)
    text=re.sub(r'\/',' ',text)
    text=re.sub(r'\-',' ',text)
    text=re.sub(r'\-+','',text) #--
    text=re.sub(r'btw','by the way',text)
    text=re.sub(r'\bla\b','',text) #la la la
    text=re.sub(r'<.*?>','',text) #<Fgfdrghd>
    text=re.sub(r'\[','',text) 
    text=re.sub(r'\]','',text)
    text=re.sub(r'hmm+','',text) #hmmmm
    text=re.sub(r'mmm+','',text) #mmmmm
    text=re.sub(r's\*\*\*\*-storm','shit-storm',text)
    text=re.sub(r'p\*\*\*y','pussy',text)
    text=re.sub(r'f\*cks','fucks',text)
    text=re.sub(r'f\*\*kn','fucking',text)
    text=re.sub(r'a\S*\**\S*\*hole','asshole',text)
    text=re.sub(r'a\S*\**\S*\*e','asshole',text)
    text=re.sub(r'c\*\*nt','cunt',text)
    text=re.sub(r'moth\S+\*+s','mother fuckers',text)
    text=re.sub(r'moth\S+\*+r','mother fucker',text)
    text=re.sub(r'b\*tch','bitch',text)
    text=re.sub(r'n\*gger','nigger',text)
    text=re.sub(r'b\*+d','bastard',text)
    text=re.sub(r'd\*vilish','devilish',text)
    text=re.sub(r'sh\*t','shit',text)
    text=re.sub(r'f\*+n','fucking',text)
    text=re.sub(r'fuckin','fucking',text)
    text=re.sub(r'dipsh\*t','dipshit',text)
    text=re.sub(r'motha\*+','mother fucking',text)
    text=re.sub(r'stoopiidd','stupid',text)
    text=re.sub(r'p\*+s','pussies',text)
    text=re.sub(r'd\*ck','dick',text)
    text=re.sub(r'f\**\S*\**face','fuck face',text)
    text=re.sub(r'lo+ng','long',text)
    text=re.sub(r'sh\*\*pile','shitpile',text)
    text=re.sub(r'b\*lls','balls',text)
    text=re.sub(r'pr\*ck','prick',text)
    text=re.sub(r'bats\*\*t','batshit',text)
    text=re.sub(r'smfh','shaking my flipping head',text)
    text=re.sub(r'\'em','asshole',text)
    text=re.sub(r'h\*ll','hell',text)
    text=re.sub(r'gtfo','get the fuck off',text)
    text=re.sub(r'fellas','fellows',text)
    text=re.sub(r'tw\*t','twat',text)
    text=re.sub(r'wh\*re','whore',text)
    text=re.sub('lgbtq','lesbian gay bisexual transgender queer',text)
    text=re.sub(r'imho','in my humble opinion',text)
    text=re.sub(r'f\*\&\^\?','fuck',text)
    text=re.sub(r'(\w)\1{3,}',r'\1',text)   
    text=re.sub(r'[\!\"\#\$\%\&\\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~]','',text)
    text=re.sub(r'[0-9]','',text)
    text=re.sub(r'\s\w\s',' ',text)
    text=re.sub(r'lol','laugh out loud',text)
    text=re.sub(r'\sfu\s','fuck you',text)
    
    stop_words = set(stopwords.words('english'))
    words=[word for word in text.split() if word not in stop_words]
    text=str((' ').join(words))
    token=spcy(text)
    text=" ".join([token.lemma_ for token in token])
    return text.strip()

pos_dic = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

def get_subjectivity(text):
    try:
        textblob = TextBlob(unicode(text, 'utf-8'))
        subj = textblob.sentiment.subjectivity
    except:
        subj = 0.0
    return subj

def pos_check(x, flag):
    x = re.sub("[^a-zA-Z]", " ", x)
    x = " ".join(x.split())
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_dic[flag]:
                cnt += 1
    except:
        pass
    return cnt


###################################################


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    x = preprocess_text(to_predict_list['Comment_text'])
    f=[]
    f.append(len(x))
    f.append(len(x.split()))
    f.append(np.mean([len(x) for x in x.split()]))
    f.append(len(re.findall(r'[A-Z]',x)))
    f.append(len(re.findall(r'\!',x)))
    f.append(len(str(x).split(".")))
    f.append(np.mean([len(x) for x in x.split('.')]))
    f.append(get_subjectivity(x))
    f.append(pos_check(x, 'noun'))
    f.append(pos_check(x, 'verb'))
    f.append(pos_check(x, 'adj'))
    f.append(pos_check(x, 'adv'))
    f.append(pos_check(x, 'pron'))
    f=np.array(f).reshape(1, -1)
    normalizer=Normalizer()
    f=normalizer.fit_transform(f)
    clf = joblib.load('NB_classifier.pickle')
    count_vect = joblib.load('vetorizer.pickle')
    tfidf=count_vect.transform([x])
    X = hstack((f, tfidf)).tocsr()
    pred = clf.predict_proba(X)
    print(pred[0])
    if pred[0][1]>=0.089:
        prediction = "Toxic"
    else:
        prediction = "Non Toxic"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
