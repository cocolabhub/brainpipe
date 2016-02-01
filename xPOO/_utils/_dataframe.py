import pandas as pd
from warnings import warn
import numpy as n


def pdKeep(df, *keep, indCol=True):
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
        indexCol.extend(list(indSet))
        dfKeep = dfKeep.append(df.iloc[list(indSet)])
    if indCol:
        dfKeep['indKeep'] = indexCol

    return dfKeep.set_index([list(n.arange(dfKeep.shape[0]))]), index
