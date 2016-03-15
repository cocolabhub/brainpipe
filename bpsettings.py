import os
import brainpipe
import pickle
import datetime
import numpy as n

__all__ = [
    'get_cfg', 'backup_cfg', 'update_cfg',
    'get_studyList', 'update_studyList', 'check_key'
]


def get_cfg(studyName):
    bpPath = os.path.dirname(brainpipe.__file__)
    with open(bpPath+'/bpsettings.pickle', "rb") as f:
        bpsettings = pickle.load(f)
    with open(bpsettings[studyName]['path']+studyName+'_settings.pickle', "rb") as f:
        studyCurrent = pickle.load(f)

    return studyCurrent


def backup_cfg(studyName):
    data = get_cfg(studyName)
    now = datetime.datetime.now()
    studySaveName = data['study_s']['path']+'backup/backup_'+studyName+'_'
    nowDate = str(now.year)+'-'+str(now.month)+'-'+str(now.day) + \
        '_'+str(now.hour)+'h'+str(now.minute)+'min'+str(now.second)+'s'

    with open(studySaveName+nowDate+'.pickle', 'wb') as f:
        pickle.dump(data, f)
    print('A backup for "'+studyName+'" has been created !')


def update_cfg(studyName, *args, backup=True):

    # Create a backup :
    if backup:
        backup_cfg(studyName)  # , print('A backup has been created !')

    # Get the cfg file of the current study :
    curCfg = get_cfg(studyName)

    for k in range(0, len(args)):
        categorie = args[k][0]
        if check_key(curCfg, categorie):
            param = args[k][1]
            paramList = list(param.keys())
            for i in paramList:  # range(0,len(param)):
                if check_key(curCfg[categorie], i):
                    print('In "', categorie, '", "', i, '"', 'has been updated from', curCfg[
                          categorie][i], 'to', param[i])
                    curCfg[categorie][i] = param[i]
                else:
                    print('Warning: in "'+categorie+'" no "'+i +
                          '" found. Here is the list of avaible keys: \n   '+str(list(curCfg[categorie].keys())))
        else:
            print('Warning: No category "'+categorie +
                  '" found in the configuration file. Here is the avaible category: \n   '+str(list(curCfg.keys())))

    print(curCfg['study_s']['path']+studyName+'_settings.pickle')
    with open(curCfg['study_s']['path']+studyName+'_settings.pickle', 'wb') as f:
        pickle.dump(curCfg, f)

    return curCfg


def get_studyList():
    bpPath = os.path.dirname(brainpipe.__file__)
    bpCfg = bpPath+'/bpsettings.pickle'
    with open(bpCfg, "rb") as f:
        bpsettings = pickle.load(f)

    return bpsettings


def update_studyList():
    bpPath = os.path.dirname(brainpipe.__file__)
    bpCfg = bpPath+'/bpsettings.pickle'
    with open(bpCfg, "rb") as f:
        bpsettings = pickle.load(f)

    studyList = list(bpsettings.keys())

    for k in studyList:
        if not os.path.exists(bpsettings[k]['path']):
            bpsettings.pop(k, None)

    with open(bpCfg, 'wb') as f:
        pickle.dump(bpsettings, f)

    return bpsettings


def check_key(dico, key):
    try:
        dico[key]
        retBool = True
    except:
        retBool = False

    return retBool


def get_name(studyName):
    cfg = get_cfg(studyName)
    name = []
    for k in range(0, cfg['subjects_s']['nsuj']):
        name.append(cfg['subjects_s']['s'+str(k+1)]['name'] +
                    '-'+cfg['subjects_s']['s'+str(k+1)]['hand'])

    return name


def load(studyName, folder, file):
    cfg = get_cfg(studyName)
    with open(cfg['study_s']['path']+folder+'/'+file, "rb") as f:
        data = pickle.load(f)
    return data


def get_fileList(studyName, folder, *args, case='lower'):
    cfg = get_cfg(studyName)
    path = cfg['study_s']['path']+folder+'/'

    ListFeat = os.listdir(path)

    if args == ():
        return ListFeat
    else:
        filterFeat = n.zeros((len(args), len(ListFeat)))
        for k in range(0, len(args)):
            for i in range(0, len(ListFeat)):
                # Case of lower case :
                if case == 'lower':
                    strCmp = ListFeat[i].lower().find(args[k].lower()) != -1
                else:
                    strCmp = ListFeat[i].find(args[k]) != -1
                if strCmp:
                    filterFeat[k, i] = 1
                else:
                    filterFeat[k, i] = 0
    return [ListFeat[k] for k in n.where(n.sum(filterFeat, 0) == len(args))[0]]
