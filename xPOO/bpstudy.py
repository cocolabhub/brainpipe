import os
import brainpipe
import pickle
from shutil import rmtree
import numpy as n
from scipy.io import loadmat


class study(object):
    """Create and manage a study with a files database.

    Parameters
    ----------
    name : string, optional [def : None]
        Name of the study. If this study already exists, this will load
        the path with the associated database.

    Methods
    ----------
    -> add_study(path) : add a new study located in path and automatically
       generate a list of folders that will be use to analyse data.

    -> delete_study() : delete the current study and all the database.

    -> file(*filters, folder='', lower=True) : return the list of files.
       Possibility filters, to define a folder for searching and comparing
       and to search in the database corresponding files in lower case.

    -> load(folder, file) : load the "file" in the "folder" location. This
       function can load .pickle or .mat files.

    -> studies() : get the list of all studies

    -> update() : update the list of all studies

    Exemple
    ----------
    # Define variables :
    path = '/home/Documents/database'
    studyName = 'MyStudy'

    # Create a study object :
    studObj = study(name)
    studObj.add_study(path) # Create the study
    studObj.studies()       # Print the list of studies

    # Manage files in your study :
    fileList = studObj.file('filter1', 'filter2', folder='features')
    # Let say that fileList contain two files : ['file1.mat', 'file2.pickle']
    # Load the second file :
    data = studObj.load('features', 'file2.pickle')



    """

    def __init__(self, name=None):
        if name:
            self.name = name
            _check_bpsettings_exist()
            _check_study_exist(self)
            _update_bpsettings()

    # -------------------------------------------------------------
    # Manage Files:
    # -------------------------------------------------------------
    def add_study(self, path):
        """Create a new study

        Parameters
        ----------
        path : string
            path to your study.

        The following folders are going to be created :
            - name : the root folder of the study. Same name as the study
            - /database : datasets of the study
            - /features : features extracted from the diffrents datasets
            - /classified : classified features
            - /multifeatures : multifeatures files
            - /figure : figures of the study
            - /physiology : physiological informations
            - /backup : some backup files
            - /settings : save some settings
        """
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
        _bpfolders(path+self.name+'/settings')
        # Add the study to the bpsetting file:
        _add_bpsettings_entry(name, path)
        _update_bpsettings()

    def delete_study(self):
        """Delete the current study
        """
        try:
            rmtree(self.path)
        except:
            print('No folder found')
        _update_bpsettings()

    # -------------------------------------------------------------
    # Manage Files:
    # -------------------------------------------------------------
    def file(self, *args, folder='', lower=True):
        """Get a list of files

        Parameters
        ----------
        *args : string, optional
            Add some filters to get a restricted list of files,
            according to the defined filters

        folder : string, optional [def : '']
            Define a folder to search. By default, no folder is specified
            so the search will focused on the root folder.

        lower : bool, optional [def True]
            Define if the search method have to take care of the case. Use
            False if case is important for searching.

        Return
        ----------
        A list containing the files found in the folder.
        """

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
        """Load a file. The file can be a .pickle or .mat

        Parameters
        ----------
        folder : string
            Specify where the file is located

        file : string
            the complete name of the file

        Return
        ----------
        A dictionary containing all the variables.
        """
        fileToLoad = self.path+folder+'/'+file
        fileName, fileExt = os.path.splitext(fileToLoad)
        if fileExt == '.pickle':
            with open(fileToLoad, "rb") as f:
                data = pickle.load(f)
        elif fileExt == '.mat':
            data = loadmat(fileToLoad)

        return data

    def save(self, type, ):
        print('To do')

    # -------------------------------------------------------------
    # Static methods :
    # -------------------------------------------------------------
    @staticmethod
    def studies():
        """Get the list of all defined studies
        """
        with open(bpCfg, "rb") as f:
            bpsettings = pickle.load(f)
        _update_bpsettings()
        print(bpsettings)

    @staticmethod
    def update():
        """Update the list of studies
        """
        _update_bpsettings()


def __bpfolders(directory):
    """Check if a folder exist otherwise, create it
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def _check_bpsettings_exist():
    """Check if the bpsettings file exist otherwise create it
    """
    try:
        _load_bpsettings()
    except:
        with open(bpPath+'bpsettings.pickle', 'wb') as f:
            pickle.dump({}, f)


def _add_bpsettings_entry(name, path):
    """Add a study to the bpsettings file. bpsettings is a
    dictionary organized by name of study and containing path.
    """
    bpsettings = _load_bpsettings()
    bpsettings[name] = path+name+'/'
    _save_bpsettings(bpsettings)


def _path_bpsettings():
    """Get the path of bpsettings
    """
    # bpPath = os.path.dirname(brainpipe.__file__)
    bpPath = os.path.dirname(brainpipe.__file__)+'/xPOO/'
    return bpPath+'/bpsettings.pickle'


def _load_bpsettings():
    """Load the bpsettings file
    """
    bpCfg = _path_bpsettings()
    with open(bpCfg, "rb") as f:
        bpsettings = pickle.load(f)
    return bpsettings


def _save_bpsettings(bpsettings):
    """Update bpsettings. The parameter should be a
    dictionary organized as bpsettings.
    """
    bpCfg = _path_bpsettings()
    with open(bpCfg, 'wb') as f:
        pickle.dump(bpsettings, f)


def _check_study_exist(self):
    """Check if a study exist
    """
    try:
        bpsettings = _load_bpsettings()
        self.path = bpsettings[self.name]
        print('-> '+self.name+' found')
    except:
        pass


def _update_bpsettings():
    """Update bpsettings list of study by checking each folder.
    """
    bpsettings = _load_bpsettings()
    bpsettingsNew = {}
    for k, i in zip(bpsettings.keys(), bpsettings.values()):
        if os.path.exists(i):
            bpsettingsNew[k] = i
    _save_bpsettings(bpsettingsNew)
