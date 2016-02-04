import numpy as n
from brainpipe.bpsettings import get_cfg, load, get_fileList
import pandas as pd


def features_load(studyName, subject, sup, *args, organize='dict', axis=0,
                  infosize='medium', width=700, step=50):

    x, featinfo, setup = {}, pd.DataFrame(), {}
    featKind, name, phy = [], [], pd.DataFrame()
    for k in args:
        xArg, y, channel, time, featinfoArg, bandArg, phyArg, setupArg = _features_load(
            studyName, subject, sup, infosize=infosize, width=width, step=step, **k)
        x.update({k['kind']: xArg})
        featKind.extend([k['kind']]), name.extend([bandArg])  # [k['band']])
        setup.update({k['kind']: setupArg})
        featinfo = featinfo.append(featinfoArg)
        phy = phy.append(phyArg)
    featinfo = featinfo.set_index([list(n.arange(featinfo.shape[0]))])
    phy = phy.set_index([list(n.arange(phy.shape[0]))])

    # Reorganize the features:
    if organize == 'array':
        x = features_dict2array(x, featKind, name, axis=axis)
    elif organize == 'list':
        x = features_dict2list(x, featKind, name)

    # Get a complete features list:
    # featList = []
    # for k in range(0,len(featinfo)):
    #     featBand = list(featinfo['feature'])[k]+'-'+list(featinfo['band'])[k]+'-'
    #     chanList = list(featinfo['channel'])[k]
    #     featList.extend([featBand+i for i in chanList])

    return x, y, channel, time, featinfo, phy, setup


def _features_load(studyName, subject, sup, kind='pow', band='', elec=None,
                   idx=None, infosize='medium', width=700, step=50):
    # - Featinfo :
    featinfo = pd.DataFrame()

    # - Get cfg file of the current study :
    cfg = get_cfg(studyName)

    # - Get physiological info :
    phy = load(studyName, 'physiology', get_fileList(
        studyName, 'physiology', subject)[0])
    channel = list(phy['Channel'])

    # - Get the feature to load :
    fList = get_fileList(studyName, 'features', subject, sup, kind, str(width), str(step))
    print(fList)
    # - Load the data, get feat and rm feat from data :
    setup = load(studyName, 'features', fList[0])
    feat, y, time = setup['feat'], setup['y'], setup['time']
    setup.pop('feat'), setup.pop('y'), setup.pop('time'), setup.pop('channel')

    # - Select a specific band :
    if band != '':
        if type(band) != list:
            band = [band]
        xBand = {}
        for k in band:
            xBand.update({k: feat[k]})
    else:
        xBand, band = feat, list(feat.keys())

    # - Select time points :
    if idx is not None:
        # Check the type of idx:
        idx = typeCheck(idx, len(xBand))
        xIdx = {}
        xBandKeys = band  # list(xBand.keys())
        for k in range(0, len(xBand)):
            xIdx.update({xBandKeys[k]: xBand[xBandKeys[k]][:, idx[k], :]})
    else:
        xIdx, idx = xBand, ['all']*len(xBand)

    # - Select channel(s):
    if elec is not None:
        elec = typeCheck(elec, len(xIdx))
        xElec = {}
        chan = []
        xIdxKeys = band  # list(xIdx.keys())
        for k in range(0, len(xIdx)):
            xElec.update({xIdxKeys[k]: xIdx[xIdxKeys[k]][elec[k], :, :]})
            if type(elec[k]) == int:
                elecLs = [elec[k]]
            else:
                elecLs = elec[k]
            chan.append([channel[x] for x in elecLs])
    else:
        # ['all']*len(xIdx), ['all']*len(xIdx)
        xElec, elec, chan = xIdx, [
            list(n.arange(0, len(channel)))]*len(idx), [channel]*len(xIdx)
    N = len(xElec)

    # - Get a resume of features info :
    if infosize == 'small':  # size of featinfo='small'
        sujI, parI, featI, bandI, elecI, chanI, idxI = [
            subject], [sup], [kind], [band], [elec], [chan], [idx]
    if infosize == 'medium':  # size of featinfo='medium'
        sujI, parI, featI, bandI, elecI, chanI, idxI = [
            subject]*N, [sup]*N, [kind]*N, band, elec, chan, idx
    if infosize == 'large':  # size of featinfo='large'
        lenElec = [len(k) for k in elec]
        elecI, idxI, chanI, bandI = [], [], [], []
        [elecI.extend(k) for k in elec], [idxI.extend([idx[k]]*lenElec[k])
                                          for k in range(0, len(idx))]
        [chanI.extend(k) for k in chan], [
            bandI.extend([band[k]]*lenElec[k]) for k in range(0, len(idx))]
        sujI, parI, featI = [subject] * \
            len(elecI), [sup]*len(elecI), [kind]*len(elecI)

    featinfo['subject'], featinfo[cfg['study_s']['parameter']],  = sujI, parI
    featinfo['feature'], featinfo['band'] = featI, bandI
    featinfo['nelec'], featinfo['channel'], featinfo[
        'time'] = elecI, chanI, idxI

    phyT = load(studyName, 'physiology', get_fileList(
        studyName, 'physiology', subject)[0])
    num, bandCat, phy = [], [], pd.DataFrame()
    for k in range(0, len(elec)):
        num.extend(elec[k]), bandCat.extend([band[k]]*len(elec[k]))
        phy = phy.append(phyT.iloc[elec[k]])
    phy['num'], phy['band'] = num, bandCat
    phy['feature'] = [kind]*len(phy)
    phy = phy.set_index([list(n.arange(phy.shape[0]))])

    return xElec, y, channel, time, featinfo, band, phy, setup


def features_dict2array(x, featKind, name, axis):
    xF = n.array([])
    for k in range(0, len(featKind)):    # For each kind of features
        for i in range(0, len(name[k])):  # For each type of features inside
            if axis == 0:
                xF = n.hstack([xF, x[featKind[k]][name[k][i]]]) if xF.size else x[
                    featKind[k]][name[k][i]]
            elif axis == 1:
                xF = n.vstack([xF, x[featKind[k]][name[k][i]]]) if xF.size else x[
                    featKind[k]][name[k][i]]
    return xF


def features_dict2list(x, featKind, name,):

    xF = []
    for k in range(0, len(featKind)):    # For each kind of features
        xF.extend([x[featKind[k]][name[k][i]] for i in range(0, len(name[k]))])
    return xF


def typeCheck(x, L):
    if type(x) == int:
        x = [[x]]*L
    # Specific case where len(x)==L:
    getType = [type(k) for k in x]
    if (len(x) == L) and (getType == [int]*len(x)):
        x = [x]*L
    if (len(x) != L) and (type(x[0]) == int):
        x = [x]*L

    x = [[k] if type(k) is int else k for k in x]

    return x
