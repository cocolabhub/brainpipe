import pandas as pd
from warnings import warn
import numpy as n

__all__ = ['pdSearch', 'pdKeep', 'pdRm']


def pdSearch(df, *keep):
    keyList = list(df.keys())
    # For each keep:
    index, indexCol = [], []
    for k in keep:
        if type(k) is not list:
            k = [k]
        # For each & condition:
        for iN, iS in enumerate(k):
            colName, toKeep = iS[0], iS[1]
            # Check if the column exist :
            if keyList.count(colName):
                if type(toKeep) is not list:
                    toKeep = [toKeep]
                # Then search in df:
                indS = []
                for jN, jS in enumerate(toKeep):
                    indS.extend(list(n.where(df[colName] == jS)[0]))
                if iN == 0:
                    indSet = set(indS)
                else:
                    indSet = indSet.intersection(set(indS))
            else:
                warn('No column "'+colName +
                     '" found. This argument has been ignored from your search'
                     )
                indSet = []

        index.append(list(indSet))

    return index


def pdKeep(df, *keep, keep_idx=True):
    """Filter a pandas dataframe and keep only interresting rows.
    This function can combine "and" and "or" conditionnal test for
    a more flexible control of a dataframe.

    df : pandas dataframe
        The dataframe to be filtered

    keep : tuple/list
        Control the informations to keep

    indCol : bool, optional [def : True]
        Add a column to df to check what are the rows that has been
        kept.

    & : inside the list
    + : for the number of arg
    Each keep: [(Column1,[s1,s2,...]),(Column2,[s21,s22])]
    """
    dfKeep = pd.DataFrame()
    keepIdx = pdSearch(df, *keep)
    indCol = []
    for k in keepIdx:
        dfKeep = dfKeep.append(df.iloc[k])
        indCol.extend(k)

    if keep_idx:
        dfKeep['keep_idx'] = indCol

    return dfKeep.set_index([list(n.arange(dfKeep.shape[0]))]), keepIdx


def pdRm(dfT, *keep, rm_idx=True):
    """Filter a pandas dataframe and keep only interresting rows.
    This function can combine "and" and "or" conditionnal test for
    a more flexible control of a dataframe.

    df : pandas dataframe
        The dataframe to be filtered

    keep : tuple/list
        Control the informations to keep

    indCol : bool, optional [def : True]
        Add a column to df to check what are the rows that has been
        kept.

    & : inside the list
    + : for the number of arg
    Each keep: [(Column1,[s1,s2,...]),(Column2,[s21,s22])]
    """
    df = dfT.copy()
    rmIdx = pdSearch(df, *keep)
    indCol = []
    for k in rmIdx:
        indCol.extend(k)
    setList = list(set(indCol))
    df.drop(setList, inplace=True)
    fullList = list(n.arange(dfT.shape[0]))

    if rm_idx:
        df['rm_idx'] = list(set(fullList).difference(setList))

    return df.set_index([list(n.arange(df.shape[0]))]), rmIdx
