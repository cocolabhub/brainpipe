import numpy as n
from pandas import DataFrame
from brainpipe.xPOO._utils._physio import loadatlas, pos2label
from brainpipe.xPOO._utils._dataframe import *


class physio(object):

    def __init__(self, atlas='tal', r=5, nearest=True, rm_unfound=False):
        self.atlas = atlas
        self.r = r
        self.nearest = nearest
        self.rm_unfound = rm_unfound
        (self.__hdr, self.__mask, self.__gray,
            self.__label) = loadatlas(atlas=self.atlas, r=self.r)

    def get(self, xyz, channel=''):
        pdPhy = pos2label(xyz, self.__mask, self.__hdr, self.__gray,
                          self.__label, r=self.r, nearest=self.nearest)
        return pd2physio(pdPhy)


class pd2physio(DataFrame):

    def __new__(self, x):
        return DataFrame.__new__(self)

    def search(self, *arg):
        return pdSearch(self, *arg)

    def keep(self, *arg, keep_idx=True):
        pdFrame, _ = pdKeep(self, *arg, keep_idx=keep_idx)
        return pd2physio(pdFrame)

    def rm(self, *arg, rm_idx=True):
        pdFrame, _ = pdRm(self, *arg, rm_idx=rm_idx)
        return pd2physio(pdFrame)

    def order(self):
        print('')

    def plot(self):
        print('')
