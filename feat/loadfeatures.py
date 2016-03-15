import numpy as n
import pandas as pd
import scipy.io as sio
import brainpipe.system.tools as tools

__all__ = [
    'load_mf', 'select_features', 'loadMultiDA'
]

####################################################################
# - Load multiple features :
####################################################################
def load_mf(sujet, main, event, band, recovery, Id,
            pathfeat='C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/',
            elec=None, timeindex=None):

    # Load one subject :
    if type(sujet) == str:
        x, y, channel, featinfo, relatedinfo = _load_mf(sujet, main, event, band, recovery, Id,
                                                        pathfeat=pathfeat, elec=elec, timeindex=timeindex)
    # Load one subject :
    elif type(sujet) == list:
        x, y, channel, featinfo, relatedinfo = [], [], [], pd.DataFrame(), pd.DataFrame()
        for k in range(0,len(sujet)):
            xT, yT, channelT, featinfoT, relatedinfoT = _load_mf(sujet[k], main[k], event, band, recovery, Id,
                                                              elec=elec, timeindex=timeindex, pathfeat=pathfeat)
            x.append(xT), y.append(yT), channel.append(channelT)
            featinfo, relatedinfo = featinfo.append(featinfoT), relatedinfo.append(relatedinfoT)
        featinfo=featinfo.set_index([list(n.arange(featinfo.shape[0]))])
        relatedinfo=relatedinfo.set_index([list(n.arange(relatedinfo.shape[0]))])

    return x, y, channel, featinfo, relatedinfo

####################################################################
# - Define the name of the feature to load :
####################################################################
def featname_fcn(sujet, main, x, band, recovery, Id):
    return sujet + '_' + main + '_event' + str(x) + '_' + band + '_' + recovery + '_Id' + Id


####################################################################
# - Simple function to load one event pear feature :
####################################################################
def load_one_event_feat(feat_path, feat_name):
    mat = sio.loadmat(feat_path + feat_name)
    feat = mat['FEAT']
    channel = mat['channelb']
    return feat, channel[0]


####################################################################
# - Load a single feature :
####################################################################
def load_sf(sujet, main, event, band, recovery, Id, 
            pathfeat='C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/',
            elec=None, timeindex=None):
    
    relatedinfo = pd.DataFrame({'Sujet':sujet, 'Main':main, 'Event':event})
    # - Define the path where features are located :
    feat_path = pathfeat + sujet + '/Features/'
    # - Load the feature :
    x = n.array([])
    y = n.array([])
    flist = []
    for k in event:
        fname = featname_fcn(sujet, main, k, band, recovery, Id)
        feat, channelT = load_one_event_feat(feat_path, fname)
        x = n.vstack([x,feat]) if x.size else feat
        C = k*n.ones((feat.shape[0],1))
        y = n.vstack([y,C]) if y.size else C
        flist.append(fname)
    relatedinfo['Features'] = flist
    relatedinfo['Timeindex'] = [timeindex]*len(flist)
    # - Reformat the channels in a list :
    channel = []
    nbchannel = len(channelT)
    for k in range(0,nbchannel):
        channel.extend(channelT[k])
    # - Save the info in a pandas structure :
    idextend = tools.extendlist([[Id]],len(channel))
    bandextend = tools.extendlist([[band]],len(channel))
    featinfo = pd.DataFrame({'Channel':channel, 'Id':idextend, 'Band':bandextend, 'Num':n.array(range(0,len(channel)))})
    # - Specific time index :
    if timeindex is not None:
        x = x[:,:,timeindex]
    # - Specific electrode :
    if elec is not None:
        x = x[:,elec]
        featinfo = featinfo.loc[elec,:]

    
    return x, n.ravel(y), channel, featinfo, relatedinfo


def _load_mf(sujet, main, event, band, recovery, Id,
            pathfeat='C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/',
            elec=None, timeindex=None):

    # - Case elec=None:
    if elec is None:
        elec = [None]*len(Id)
    if len(elec) is not len(Id):
        elec = [elec]*len(Id)
    # - Case timeindex=None:
    if timeindex is None:
        timeindex = [None]*len(Id)
    if len(timeindex) is not len(Id):
        timeindex = [timeindex]*len(Id)
    # - Load multiple features :
    featinfo = pd.DataFrame()
    relatedinfo = pd.DataFrame()
    xmf = n.array([])
    for k in range(0,len(Id)):
        xsf, y, channel, featinf, relatedinf = load_sf(sujet, main, event, band[k], recovery[k], Id[k], 
                                          pathfeat, elec=elec[k], timeindex=timeindex[k])
        xmf = n.hstack([xmf,xsf]) if xmf.size else xsf
        featinfo = featinfo.append(featinf)
        relatedinfo = relatedinfo.append(relatedinf)
        del xsf, featinf, relatedinf
    # - Set the correct index :
    featinfo=featinfo.set_index([list(n.arange(featinfo.shape[0]))])
    relatedinfo=relatedinfo.set_index([list(n.arange(relatedinfo.shape[0]))])

    return n.squeeze(xmf), y, channel, featinfo, relatedinfo


####################################################################
# - Select features in a pandas structure :
####################################################################
def select_features(featinfo, group, select):
    X = featinfo.groupby(group)
    return X.groups[select]

#__________________________________________________________________________________________________________________
#__________________________________________________________________________________________________________________

####################################################################
# - Load multiple DA :
####################################################################
def loadMultiDA(sujet, main, band, event, recovery, Id, classifier, elec=None, timeindex=None, condition='',
                permutation=None, pathfeat='C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/'):
    # Load one subject :
    if type(sujet) == str:
        x, xperm, y, channel, featinfo, relatedinfo = _loadMultiDA(sujet, main, band, event,
                                                                   recovery, Id, classifier, elec=elec,
                                                                   timeindex=timeindex, condition=condition,
                                                                   permutation=permutation, pathfeat=pathfeat)
    # Load one subject :
    elif type(sujet) == list:
        x, xperm, y, channel, featinfo, relatedinfo = [], [], [], [], pd.DataFrame(), pd.DataFrame()
        for k in range(0,len(sujet)):
            # Load one subject :
            xT, xpermT, yT, channelT, featinfoT, relatedinfoT = _loadMultiDA(sujet[k], main[k], band,
                                                                             event, recovery, Id, classifier, elec=elec,
                                                                       timeindex=timeindex, condition=condition,
                                                                       permutation=permutation, pathfeat=pathfeat)

            x.append(xT), xperm.append(xpermT), y.append(yT), channel.append(channelT)
            featinfo, relatedinfo = featinfo.append(featinfoT), relatedinfo.append(relatedinfoT)
        # Reset index :
        featinfo=featinfo.set_index([list(n.arange(featinfo.shape[0]))])
        relatedinfo=relatedinfo.set_index([list(n.arange(relatedinfo.shape[0]))])

    return x, xperm, y, channel, featinfo, relatedinfo

def _loadMultiDA(sujet, main, band, event, recovery, Id, classifier, elec=None, timeindex=None, condition='',
                 permutation=None, pathfeat='C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/'):
    # First, load info :
    _, y, channel, featinfo, relatedinfo = load_mf(sujet, main, event, band, recovery, Id,  elec=elec,
                                                   timeindex=timeindex, pathfeat=pathfeat)
    # - Case elec=None:
    nbfeat = len(band)
    if elec is None:
        elec = [None]*nbfeat
    if len(elec) is not nbfeat:
        elec = [elec]*nbfeat
    # - Case timeindex=None:
    if timeindex is None:
        timeindex = [None]*nbfeat
    if len(timeindex) is not nbfeat:
        timeindex = [timeindex]*nbfeat
    # Load the DA :
    x, xperm, tidx = n.array([]), n.array([]), []
    for k in range(0,nbfeat):
        DA, DAperm = loadOneDA(sujet, main, band[k], recovery[k], Id[k], classifier, pathfeat=pathfeat,
                elec=elec[k], timeindex=timeindex[k], condition=condition[k], permutation=permutation)
        x = n.hstack([x,DA]) if x.size else DA
        xperm = n.hstack([xperm,DAperm]) if xperm.size else DAperm
    if xperm.shape[0] == 1: x, xperm = x[0], xperm[0]
    # Set DA to featinfo and
    featinfo['Decoding accuracy (%)'], featinfo['Permutation'] = x, xperm
    featinfo=featinfo.set_index([list(n.arange(featinfo.shape[0]))])

    return x, xperm, y, channel, featinfo, relatedinfo

def loadOneDA(sujet, main, band, recovery, Id, classifier,
            pathfeat='C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/',
            elec=None, timeindex=None, condition='', permutation=None):
    # - Define the path where features are located :
    feat_path = pathfeat + sujet
    # Load the decoding accuracy :
    if permutation == None:
        name = sujet+'_'+main+'_'+band+condition+'_'+recovery+'_Id'+Id+'_'+classifier
        DA = sio.loadmat(pathfeat + sujet + '/Classified/' + name+'.mat')['ALL']
        DAperm = n.zeros(DA.shape)
    else:
        name = sujet+'_'+main+'_'+band+condition+'_'+recovery+'_Id'+Id+'_Perm_'+classifier
        mat = sio.loadmat(pathfeat + sujet + '/Permutations/' + name+'.mat')
        DA, DAperm = mat['ALL'], mat['DAperm']
    # Keep the selected time index and electrode :
    if elec is not None:
        DA, DAperm = DA[:,elec], DAperm[:,elec]
    if timeindex is not None:
        DA, DAperm = DA[timeindex,:], DAperm[timeindex,:]
    return DA, DAperm

