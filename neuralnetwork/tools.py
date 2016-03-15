import numpy as n

####################################################################
# - Transform labels to compatible binay NN :
####################################################################
def label2binaire(y):
    yunique = n.unique(y)
    nbtrials = len(y)
    nbclass = len(yunique)
    ybinaire = n.zeros((nbtrials,nbclass))
    for k in range(0,nbclass):
        ybinaire[y==yunique[k],k] = 1
    return ybinaire

####################################################################
# - Transform binay NN to labels :
####################################################################
def binaire2label(ybinaire):
    nbtrials, nbclass = ybinaire.shape
    y = n.zeros((nbtrials,1))
    for k in range(0,nbclass):
        y[ybinaire[:,k]==1,0]=k
    y = n.array(y).T[0]
    return list(y.astype(int))

####################################################################
# - Eval binay NN predictions :
####################################################################
def evalbianypred(ypred,y):
    nbtrials, nbclass = y.shape
    eq = y==ypred
    scorebin = n.sum(eq.astype(int),1)
    scorebin[scorebin<nbclass] = 0
    scorebin[scorebin==nbclass] = 1
    score = scorebin.sum()/nbtrials
    return score


