##############################################
#  Reddit utilities
#  (To be documented further in the future)
##############################################

##############################################
# Packages
##############################################
import praw
import pandas as pd
import datetime as dt
import os
import re

from html.parser import HTMLParser
import random as rnd

##############################################
# Define external handlers (Reddit, TreeText)
##############################################
REDDIT = praw.Reddit(client_id=os.getenv('reddit_client_id'), \
                     client_secret=os.getenv('reddit_client_secret'), \
                     user_agent=os.getenv('reddit_user_agent'), \
                     username='', \
                     password='')

##############################################
# TreeTagger
##############################################

with open(os.getenv('python_utilities') + 'tree_tagger.py') as fd:
    exec(fd.read())

##############################################
# Submission archive handling
##############################################

def get_date(created):
    return dt.datetime.fromtimestamp(created)

class eah_submission_archive:
    def __init__(self, df):
        tp = type(df).__name__ 
        if tp == 'DataFrame':
            self.data = df
        elif tp == 'str':
            self.data = pd.read_pickle(df)
        
    def save(self, filen=None):
        if filen is None:
            ts = str(dt.datetime.now()).replace(':','-').replace(' ','-').replace('.','-')
            filen = 'subm' + ts + '.pkl'
        self.data.to_pickle(filen)
        
class eah_submission_collector:
    def __init__(self, subname, limit=25, top=False):
        self.subredditName = subname
        self.subreddit = REDDIT.subreddit(subname)
        sdict = { "id" : [], "created" : [], "title": [], "url" : []}
        
        if top:
            nn = self.subreddit.top(limit=limit)
        else:
            nn = self.subreddit.new(limit=limit)
        for submission in nn:
            sdict["id"].append(submission.id)
            sdict["created"].append(submission.created)
            sdict["title"].append(submission.title)
            sdict["url"].append(submission.url)
        sdata = pd.DataFrame(sdict)
        sdata["timestamp"] = sdata["created"].apply(get_date)
        sdata.set_index(sdata['id'], inplace=True)
        self.data = eah_submission_archive(sdata)
        
    def fullSubmission(self, i):
        return(REDDIT.submission(id=self.data["id"][i]))
    
    def save(self, filen):
        self.data.save(filen)
        
    def ids(self):
        return(self.data.data['id'])

def eah_resolve_submissions(subList):
    id = []
    for s in subList:
        id = id + list(s.data['id'])
    ii = list(set(id))
    ii.sort()
    
    o = {}
    for ss in ii:
        o[ss] = []
    for s in subList:
        for ss in list(s.data['id']):
            o[ss].append(s)
    return(o)

def eah_get_fullSubmission(keys, verbose=False):
    o = []
    for k in keys:
        if verbose:
            print(k)
        s = REDDIT.submission(id=k)
        s.comments.replace_more(limit=None)
        o.append(s)
    return(o)

def eah_find_archives(search='^subm([a-zA-Z-0-9]*)*[.]pkl$'):
    o = []
    for s in os.listdir():
        if len(re.findall(search,s))>0:
            o.append(s)
    return(o)

##############################################
# Comment retrieval
##############################################

def eah_get_comments(subList, html=1):
    d_sub = []
    d_id = []
    d_aut = []
    d_body = []
    d_score = []
    d_crt = []
    d_shid = []
    for sub in subList:
        subid = str(sub.id)
        cms = sub.comments.list()
        for cm in cms:
            d_sub.append(subid)
            d_id.append(cm.id)
            d_score.append(cm.score)
            d_crt.append(cm.created)   
            d_shid.append(cm.score_hidden)
            if cm.author is None:
                d_aut.append('')
            else:
                d_aut.append(cm.author.name)
            if html==1:
                d_body.append(cm.body_html)
            else:
                d_body.append(cm.body)
                
    o = pd.DataFrame()
    o['subm_id'] = d_sub
    o['comm_id'] = d_id
    o['author'] = d_aut
    o['score'] = d_score
    o['body'] = d_body
    o['created'] = d_crt
    o['score_hidden'] = d_shid
    
    o.set_index(o['comm_id'], inplace=True)
    
    return(o)

##############################################
# Comment parsing
##############################################

class RedditHTMLParser(HTMLParser):
    def __init__(self, capture=['a', 'em', 'strong']):
        HTMLParser.__init__(self)
        self.capture = capture
        self.text = []
        self.href = []
        self.tags = []
        self.state = {}
        self.state_href = ''

        for tg in capture:
            self.state[tg] = 0        
        
    def handle_starttag(self, tag, attrs):
        for tg in self.capture:
            if tag==tg:
                self.state[tag] = 1
                n = len(attrs)
                if n>0:
                    for i in range(n):
                        if attrs[i][0]=='href':
                            self.state_href = attrs[i][1]
        
    def handle_endtag(self, tag):
        for tg in self.capture:
            if tag==tg:
                self.state[tag] = 0

    def handle_data(self, data):
        self.text.append(data)
        self.tags.append(self.state.copy())
        
        self.href.append(self.state_href)
        self.state_href = ''
            
    def DataFrame(self):
        df = pd.DataFrame()
        df['text'] = self.text
        
        tags = {}
        for tg in self.capture:
            tags[tg] = []
        
        for i in range(len(self.text)):
            for tg in self.capture:
                tags[tg].append(self.tags[i][tg])

        for tg in self.capture:
            df[tg] = tags[tg]            

        df['href'] = self.href
        
        return(df)
    
    def CleanText(self, sep =''):
        out = ''
        for s in self.text:
            out += (sep + s)
        return(out)

def eah_parse_comments(cmdb):
    n = cmdb.shape[0]
    data = {}
    text = {}
    for i in range(n):
        cid = cmdb.iloc[i]['comm_id']
        parser = RedditHTMLParser()
        parser.feed(cmdb.iloc[i]['body'])
        data[cid] = parser.DataFrame()
        text[cid] = parser.CleanText()
    return({'data':data, 'clean_text':text})

def eah_tag_comments(pcmdd, keys = None):
    if keys is None:
        keys = list(pcmdd['clean_text'].keys())
    ttxt = treeTagAllText([pcmdd['clean_text'][k] for k in keys])
    for i in range(len(keys)):
        ttxt[i]['id'] = [keys[i]]*len(ttxt[i])
    return(pd.concat(ttxt)) 

##############################################
# Other utilities
##############################################

def matrixToDataFrame(X,f):
    D = pd.DataFrame()
    J = X.shape[1]
    N = X.shape[0]
    for i in range(J):
        D[f[i]] = np.asarray(X[:N,i],dtype='float').reshape(N)
    return(D)

class randomPartition:
    def __init__(self, i, nfolds=10):
        self.nfolds = nfolds
        self.folds = list(range(nfolds))
        self.n = len(i)
        i = np.array(i)
        s = np.array(rnd.choices(self.folds, k=self.n))
        o = []
        for f in self.folds:
            o.append(i[s==f])
        self.index = o
        
    def fold(self, fi):
        return(self.index[fi])
    
    def foldComplement(self, fi):
        o = []
        for i in range(self.nfolds):
            if i != fi:
                o = o + list(self.index[i])
                
        return(o)    
    
    def crossValidate(self, fun):
        o = []
        for i in range(self.nfolds):
            o.append(fun(self.fold(i),self.foldComplement(i)))
        return(o)
    
    def weights(self):
        return(np.array([len(i) for i in self.index])/self.n)