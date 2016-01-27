import os
import brainpipe
import pickle
from shutil import rmtree
import numpy as n


class study(object):

    def __init__(self, name=None):
        if name:
            self.name = name
            _check_bpsettings_exist()
            _check_study_exist(self)
            _update_bpsettings()

    def add(self, path):
        # Main studyName directory :
        _bpfolders(path+self.name)
        # Subfolders :
        _bpfolders(path+self.name+'/database')
        _bpfolders(path+self.name+'/features')
        _bpfolders(path+self.name+'/classified')
        _bpfolders(path+self.name+'/multifeatures')
        _bpfolders(path+self.name+'/figures')
        _bpfolders(path+self.name+'/backup')
        _bpfolders(path+self.name+'/physiology')
        # Add the study to the bpsetting file:
        _add_bpsettings_entry(name, path)
        _update_bpsettings()

    def delete(self):
        try:
            rmtree(self.path)
        except:
            print('No folder found')
        _update_bpsettings()

    def file(self, folder, *args, lower=True):

        ListFeat = os.listdir(self.path+folder+'/')

        if args == ():
            return ListFeat
        else:
            filterFeat = n.zeros((len(args), len(ListFeat)))
            for k in range(0, len(args)):
                for i in range(0, len(ListFeat)):
                    # Case of lower case :
                    if lower:
                        strCmp = ListFeat[i].lower().find(
                            args[k].lower()) != -1
                    else:
                        strCmp = ListFeat[i].find(args[k]) != -1
                    if strCmp:
                        filterFeat[k, i] = 1
                    else:
                        filterFeat[k, i] = 0
        return [ListFeat[k] for k in n.where(n.sum(filterFeat, 0) == len(
                    args))[0]]

    def load(self, folder, file):
        with open(self.path+folder+'/'+file, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def studies():
        with open(bpCfg, "rb") as f:
            bpsettings = pickle.load(f)
        _update_bpsettings()
        print(bpsettings)

    @staticmethod
    def update():
        _update_bpsettings()


def __bpfolders(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _check_bpsettings_exist():
    try:
        _load_bpsettings()
    except:
        with open(bpPath+'bpsettings.pickle', 'wb') as f:
            pickle.dump({}, f)


def _add_bpsettings_entry(name, path):
    bpsettings = _load_bpsettings()
    bpsettings[name] = path+name+'/'
    _save_bpsettings(bpsettings)


def _path_bpsettings():
    # bpPath = os.path.dirname(brainpipe.__file__)
    bpPath = os.path.dirname(brainpipe.__file__)+'/xPOO/'
    return bpPath+'/bpsettings.pickle'


def _load_bpsettings():
    bpCfg = _path_bpsettings()
    with open(bpCfg, "rb") as f:
        bpsettings = pickle.load(f)
    return bpsettings


def _save_bpsettings(bpsettings):
    bpCfg = _path_bpsettings()
    with open(bpCfg, 'wb') as f:
        pickle.dump(bpsettings, f)


def _check_study_exist(self):
    try:
        bpsettings = _load_bpsettings()
        self.path = bpsettings[self.name]
        print('-> '+self.name+' found')
    except:
        pass


def _update_bpsettings():
    bpsettings = _load_bpsettings()
    bpsettingsNew = {}
    for k, i in zip(bpsettings.keys(), bpsettings.values()):
        if os.path.exists(i):
            bpsettingsNew[k] = i
    _save_bpsettings(bpsettingsNew)
