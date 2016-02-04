import numpy as n
import pandas as pd
from brainpipe.xPOO._utils._physio import loadatlas, pos2label


class physio():

    def __init__(self, atlas='tal', r=5, nearest=True):
        self.atlas = atlas
        self.r = r
        self.nearest = nearest
        (self.__hdr, self.__mask, self.__gray, self.__brodtxt,
            self.__brodidx, self.__label) = loadatlas(atlas=atlas)

    def get(self, xyz, channel=''):
        print('')


class pd2physio(pd.Dataframe):

    def __new__(self, x):
        return pd.Dataframe.__new__(self, x)

    def keep():
        print('')

    def rm():
        print('')

    def order():
        print('')

    def plot():
        print('')
