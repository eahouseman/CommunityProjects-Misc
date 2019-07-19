##############################################
#  Lexical utilities
#  (To be documented further in the future)
##############################################

import os
import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt
from scipy import stats as stats
from sklearn import feature_extraction as fe
from scipy import sparse as sparse
import random 
import scipy

import seaborn
from sklearn import linear_model as lm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

##############################################

# For each x, determine if x is a member of a set S
def InSet(x,S):
    flag = False
    for s in S:
        flag = flag | (x == s)
    return(flag)

def getTermSubmatrix(X, allterms, terms):
    cflag = InSet(allterms,terms)
    xx = X[:,cflag] * np.ones(sum(cflag))
    return({'X': X[xx>0,:][:,cflag==False], 'F' : allterms[cflag==False]})

def rowNormalize(X,Xs):
    Xcount = Xs * np.ones(Xs.shape[1])
    Xcount[Xcount==0] = 1 # to avoid divide-by-zero error
    return(scipy.sparse.diags(1/Xcount) * X)

def wordSort(coef, words):
    ix = pd.DataFrame()
    ix['index'] = list(range(len(coef)))
    ix['value'] = coef
    ix['word']= words
    #ix.set_index('word')
    ix = ix.sort_values('value', ascending=False)
    return(ix)

def wordBarPlot(ix, main='', n=10, xerr=False):
    if xerr:
        plt.barh(ix[:n]['word'],ix[:n]['value'],xerr=ix[:n]['std'])
    else:
        plt.barh(ix[:n]['word'],ix[:n]['value'])
    plt.title(main)
    plt.show()

##############################################
# Vectorization
##############################################

def vectorize_tags(taglist, types = ['N','V','J'], exclude=['be','do','have','get','use'], ngram=1):
    x = []
    wds = set([])
    if ngram>1:
        print('Warning:  ngram>1 not yet implemented.')
    for tag in taglist:
        tagd = tag.copy()
        tagd['PoS'] = [x[0] for x in tagd['tag']]
        tagd['stem'] = [s.lower() for s in tagd['stem']]
        ntypes = len(types)
        flag = InSet(tagd['PoS'], types)
        flagex = InSet(tagd['stem'], exclude)
        flag = flag & (flagex == False)

        if ngram>1:
            flag = flag | (tagd['tag']=='SENT')
            # To implement ngram>1, need to combine sequential stems,
            #  but not across SENT boundaries
        
        wc = tagd[flag].groupby('stem')['stem'].count()
        wds = wds.union(list(wc.index))
        x.append(wc)
        
    nwds = len(wds)
    n = len(x)
 
    wds = list(wds)
    wds.sort()
    wix = pd.DataFrame()
    wix['index'] = list(range(nwds))
    wix['stem'] = wds
    wix.set_index('stem',inplace=True)
    X = sparse.lil_matrix((n,nwds), dtype='i')
    for i in range(n):
        ii = wix.loc[list(x[i].index)]['index']
        X[i,ii] = list(x[i])
    return({'stems' : wds, 'matrix' : X.tocsr()})

class stem_matrix:
    def __init__(self, stems, vec, filtersn=[1,10]):
        self.X = vec
        self.features = np.array(stems)
        self.nstems = self.X.shape[1]
        self.wcount = np.asarray(sum(self.X.todense())).reshape(self.nstems)

        self.filtersn = filtersn
        self.filters = []
        for fn in filtersn:
            self.filters.append(self.wcount>fn)
            
    def getX(self, i=0):
        if i==0:
            return(self.X)
        else:
            return(self.X[:,self.filters[i-1]])

    def getFeatures(self, i=0):
        if i==0:
            return(self.features)
        else:
            return(self.features[self.filters[i-1]]) 

##############################################
# Encapsulation of various statistical models
#  (To be improved upon in the future)
##############################################

class LassoPath:
    def __init__(self,X,y,features,logalphas=[-6,-5,-4],normalize=False):
        self.fits = []
        self.indices = []
        self.features = features
        self.logalphas = logalphas
        self.nalphas = len(logalphas)
        alphas = 10.0**np.array(logalphas)
        for a in alphas:
            lasso = lm.Lasso(alpha=a,normalize=normalize,positive=False)
            lasso.fit(X,y)
            self.fits.append(lasso)
            self.indices.append(wordSort(lasso.coef_, self.features))

    def coefficients(self):
        return(np.concatenate([f.coef_.reshape(f.coef_.shape[0],1) for f in self.fits],axis=1))
    
    def coefficientData(self):
        df = pd.DataFrame()
        for i in range(self.nalphas):
            df[str(self.logalphas[i])] = self.fits[i].coef_
        df.set_index(self.features, inplace=True)
        return(df)
    
    def bar(self, i, main='', n=10):
        wordBarPlot(self.indices[i],main,n)

    def pathMap(self, n=100):
        i = self.indices[0].iloc[0:n]['index']
        seaborn.clustermap(self.coefficientData().iloc[i])
         
class LassoBootPath:
    def __init__(self,X,y,features,nboot=5,logalphas=[-6,-5,-4],normalize=False):

        self.fits = []
        self.indices = []
        self.features = features
        self.logalphas = logalphas
        self.nalphas = len(logalphas)
        self.nsubj = X.shape[0]
        self.boot_indices=[]

        alphas = 10.0**np.array(logalphas)
        ix = list(range(self.nsubj))

        for i in range(nboot):
            self.boot_indices.append(random.choices(ix, k=self.nsubj))
        for a in alphas:
            lassos = []
            print(a)
            for i in range(nboot):
                lasso = lm.Lasso(alpha=a,normalize=normalize,positive=False)
                lasso.fit(X[self.boot_indices[i],:],y[self.boot_indices[i]])
                lassos.append(lasso)
            self.fits.append(lassos)
            
        cfs = self.coefficients()
        n = cfs.shape[0]
        for i in range(cfs.shape[2]):
            mu = []
            sig = []
            for j in range(n):
                mu.append(np.mean(cfs[j,:,i]))
                sig.append(np.std(cfs[j,:,i]))
            ix = wordSort(np.array(mu), self.features)
            ix['std'] = np.array(sig)[ix['index']]
            self.indices.append(ix)

    def coefficients(self):
        nboot = len(self.boot_indices)
        cfs = []
        for ff in self.fits:
            for f in ff:
                n = f.coef_.shape[0]
                x = np.concatenate([f.coef_.reshape(n,1) for f in ff],axis=1)
            cfs.append(x.reshape(n,nboot,1))   
        return(np.concatenate(cfs,axis=2))
    
    def coefficientData(self):
        df = pd.DataFrame()
        for i in range(self.nalphas):
            df[str(self.logalphas[i])] = self.fits[i].coef_
        df.set_index(self.features, inplace=True)
        return(df)
    
    def bar(self,i, main='', n=10):
        wordBarPlot(self.indices[i],main,n,xerr=True)

    def pathMap(self, n=100):
        i = self.indices[0].iloc[0:n]['index']
        seaborn.clustermap(self.coefficientData().iloc[i])
        
class LDApath:
    def __init__(self, X, features, Klist=list(range(1,10)), random_state=0):
        self.Klist = Klist
        self.features = features
        self.random_state = random_state
        self.X = X
        self.lda = []
        self.perplex = []
        self.score = []
        for k in Klist:
            lda = LatentDirichletAllocation(n_components=k,random_state=random_state)
            lda.fit(X)
            self.lda.append(lda)
            px = lda.perplexity(X)
            ll = lda.score(X)
            self.perplex.append(px)
            self.score.append(ll)
            print('K = %i, perplex = %f, log-like = %f' % (k,px,ll))
            
    def plotPerplexity(self):
        plt.plot(self.Klist, self.perplex, 'b')
        plt.xlabel('# components')
        plt.ylabel('perplexity')
        plt.show()

    def plotScore(self):
        plt.plot(self.Klist, self.score, 'b')
        plt.xlabel('# components')
        plt.ylabel('log-likelihood')
        plt.show()
        
    def posteriorWeights(self,kindex):
        return(self.lda[kindex].transform(self.X))
    
    def loadings(self,kindex):
        return(self.lda[kindex].components_)
    
    def showFeatures(self,kindex,dim,nfeatures=25):
        cmp = self.loadings(kindex)
        ws = wordSort(cmp[dim,:], self.features)
        wordBarPlot(ws,n=nfeatures)

class NMFpath:
    def __init__(self, X, features, Klist=list(range(1,10)), random_state=0, alpha=0):
        self.Klist = Klist
        self.features = features
        self.random_state = random_state
        self.X = X
        n = self.X.shape[0]
        self.mu = (np.ones(n)/n) * self.X
        self.nmf = []
        self.score = []
        for k in Klist:
            nmf = NMF(n_components=k,random_state=random_state,alpha=alpha)
            nmf.fit(X)
            self.nmf.append(nmf)
            yhat = np.matmul(nmf.transform(self.X) , nmf.components_)
            nn = np.product(yhat.shape)
            rhat = np.asarray(self.X - yhat).reshape(nn)
            ll = np.sqrt(np.sum(rhat*rhat)/nn)
            self.score.append(ll)
            print('K = %i, residual error = %f' % (k,ll))

    def plotScore(self):
        axes = plt.plot(self.Klist, self.score, 'b')
        plt.xlabel('# components')
        plt.ylabel('residual')
        return(axes)
        
    def posteriorWeights(self,kindex):
        return(self.nmf[kindex].transform(self.X))
    
    def loadings(self,kindex):
        return(self.nmf[kindex].components_)
    
    def showFeaturesAsBar(self,kindex,dim,nfeatures=25):
        cmp = self.loadings(kindex)
        ws = wordSort(cmp[dim,:]/self.mu, self.features)
        return(wordBarPlot(ws,n=nfeatures))

    def showFeaturesAsMap(self,kindex=0,n=10,epsilon=0.0001,normalize=False,figsize=None):
        cmp = self.loadings(kindex)
        wds = set([])
        nc = cmp.shape[0]
        for i in range(nc):
            x = cmp[i,:]
            if normalize:
                x = x/self.mu
            ws = wordSort(x, self.features)
            wds = wds.union(ws[:n]['word'])

        df = pd.DataFrame()
        for w in wds:
            df[w] = np.log10(cmp[:,self.features==w][:,0]+epsilon)

        df.set_index(np.array(range(nc))+1, inplace=True)
        return(seaborn.clustermap(df,z_score=1,figsize=figsize))
        
