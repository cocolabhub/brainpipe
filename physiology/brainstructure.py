import numpy as n
import scipy.io as scio
from pandas import DataFrame, concat
import os, inspect
from brainpipe.system.tools import MeltingSort, repDataFrame


def get_PhyInfo(sujet, pathdef, atlas='Talairach', r=5, nearest='on', rm_unfound=False, sortby=None, keeponly=None,
                 rm_tuple=None, rep=0):
    """Get physio"""

    if type(sujet) == str: sujet=[sujet]

    # First, load all subjects :
    nbsuj = len(sujet)
    chan_info = DataFrame()
    for k in range(0,nbsuj):
        sujchan = loadoneXYZ(sujet[k],pathdef)
        chan_info = chan_info.append(sujchan)

    # Reset the correct index :
    chan_info=chan_info.set_index([list(n.arange(chan_info.shape[0]))])

    # Load the physiological info :
    chan_xyz = chan_info[['X','Y','Z']].values
    inf = getphysiologicalinfo(chan_xyz,atlas=atlas, r=r, nearest=nearest, rm_unfound=rm_unfound)

    # Concat the two pandas structures :
    PhyCat = concat([inf,chan_info], axis=1)

    # Delete unfound lines :
    if rm_unfound == True:
        PhyCat = rm_info(PhyCat, (['Not found','No Gray Matter found within +/-'+str(r)+'mm'],['Brodmann', 'Brodmann']))

    # Delete user'defined lines :
    if rm_tuple != None: PhyCat = rm_info(PhyCat, rm_tuple)

    # Keep only some lines:
    if keeponly != None: PhyCat = keep_only(PhyCat,keeponly)

    # Sort the Dataframe :
    if sortby != None: PhyCat = physort(PhyCat, sortby=sortby)

    # Repeat the DataFrame:
    if rep != 0: PhyCat = repDataFrame(PhyCat,rep)

    return PhyCat


def loadoneXYZ(sujet,pathdef):
    sname = sujet+'_channel_bip'
    pathelec = pathdef+sujet+'/Channels/'
    # Load data:
    data = scio.loadmat(pathelec+sname)
    chan = data['channelb_bip']
    nbelec = chan.shape[0]
    # Organize data
    chan_name, chan_xyz = [], n.zeros((nbelec,3))
    for k in range(0,nbelec):
        chan_name.extend(chan[k,0][0][0])
        chan_xyz[k,0], chan_xyz[k,1], chan_xyz[k,2] = chan[k,1][0][0], chan[k,2][0][0], chan[k,3][0][0]
    pds = [sujet]*nbelec
    # Save data in a panda structure :
    chan_frame = DataFrame({'Channel':chan_name, 'X':chan_xyz[:,0], 'Y':chan_xyz[:,1], 'Z':chan_xyz[:,2], 'Sujet':pds})
    return chan_frame


def getphysiologicalinfo(pos,atlas='Talairach', r=5, nearest='on', rm_unfound=False):
    # load atlas :
    hdr, mask, gray, brodtxt, brodidx, label = loadatlas(atlas=atlas)

    # Get info of each electrode :
    nbchan = pos.shape[0]
    hemi, lobe, gyrus, matter, brod = [], [], [], [], []
    for k in range(0,nbchan):
        hemiC, lobeC, gyrusC, matterC, brodC = physiochannel(list(pos[k,:]),mask,hdr,gray,label,r=r,nearest=nearest)
        hemi.extend(hemiC), lobe.extend(lobeC), gyrus.extend(gyrusC)
        matter.extend(matterC), brod.extend(brodC)

    # Put everything in a panda structure :
    phyinf = DataFrame({'Hemisphere':hemi, 'Lobe':lobe, 'Gyrus':gyrus, 'Matter':matter, 'Brodmann':list(brod)})

    # Replace corrupted values :
    phyinf.replace(to_replace='*', value='Not found', inplace=True)
    phyinf['Brodmann'].replace(to_replace='Brodmann area ', value='', inplace=True, regex=True)
    phyinf['Gyrus'].replace(to_replace=' Gyrus', value='', inplace=True, regex=True)
    phyinf['Lobe'].replace(to_replace=' Lobe', value='', inplace=True, regex=True)
    phyinf['Matter'].replace(to_replace=' Matter', value='', inplace=True, regex=True)

    # Convert Brodmann to number :
    BrodStr = list(phyinf['Brodmann'])
    for k in range(0,len(BrodStr)):
        try:
             BrodStr[k] = int(BrodStr[k])
        except :
             BrodStr[k] = BrodStr[k]
    phyinf['Brodmann'] = BrodStr

    return phyinf


def physiochannel(pos,mask,hdr,gray,label,r=5,nearest='on'):
    pos = mni2tal(pos)
    ind = coord2ind(pos,mask,hdr,gray,r=r, nearest=nearest)
    if ind == -1:
        hemi, lobe, gyrus, matter, brod = ['Not found'], ['Not found'], ['Not found'], ['Not found'], ['Not found']
    elif ind == -2:
        hemi, lobe, gyrus, matter, brod = ['No Gray Matter found within +/-'+str(r)+'mm'], ['No Gray Matter found within +/-'+str(r)+'mm'], ['No Gray Matter found within +/-'+str(r)+'mm'], ['No Gray Matter found within +/-'+str(r)+'mm'], ['No Gray Matter found within +/-'+str(r)+'mm']
    else:
        hemi, lobe, gyrus, matter, brod = [label[ind,:][0][0]], [label[ind,:][1][0]], [label[ind,:][2][0]], [label[ind,:][3][0]], [label[ind,:][4][0]]
    return hemi, lobe, gyrus, matter, brod


def coord2ind(pos,mask,hdr,gray,r=5, nearest='on'):
    # Apply a transformation of position:
    pos.extend([1])
    sub = list(n.around(n.array(n.linalg.lstsq(hdr,n.matrix(pos).T)[0].T)[0]).astype(int))
    # Find the index with the nearest option:
    if nearest is 'off':
        try:
            ind = mask[sub[0], sub[1], sub[2]]-1
        except:
            ind = -1
    elif nearest is 'on':
        if sub[2] > gray.shape[2]:
            sub[2] = gray.shape[2]-1
        tranche = n.squeeze(gray[:,:,sub[2]])
        # Euclidian distance :
        dist = 100*n.ones((tranche.shape))
        u,v = n.where(tranche == 1)
        for k in range(0,len(u)):
            dist[u[k], v[k]] = n.math.sqrt((sub[1]-v[k])**2 + (sub[0]-u[k])**2)
        mindist = dist.min()
        umin,vmin = n.where(dist == mindist)
        if mindist < r:
            ind = mask[umin[0],vmin[0],sub[2]]-1
        else:
            ind = -2
    return ind


def loadatlas(atlas='Talairach'):
    B3Dpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # Load AAL atlas :
    if atlas is 'AAL': # PAS FINIT
        AAL = scio.loadmat(B3Dpath + '/Atlas/Labels/BrainNet_AAL_Label')
        hdr, label, mask = AAL['AAL_hdr'], AAL['AAL_label'], AAL['AAL_vol']

    # Load talairach atlas :
    if atlas is 'Talairach':
        TAL = scio.loadmat(B3Dpath + '/Atlas/Labels/Talairach_atlas')
        hdr = TAL['hdr']['mat'][0][0]

        label, mask, gray = TAL['label'], TAL['mask'], TAL['gray']
        brodtxt, brodidx = TAL['brod']['txt'][0][0], TAL['brod']['idx'][0][0]

    return hdr, mask, gray, brodtxt, brodidx, label

def spm_matrix(P):


    q  = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    P.extend(q[len(P):12])

    T = n.matrix([[1, 0, 0, P[0]], [0, 1, 0, P[1]], [0, 0, 1, P[2]], [0, 0, 0, 1]])
    R1 = n.matrix([ [1, 0, 0, 0], [0, n.cos(P[3]), n.sin(P[3]), 0],
                   [0, -n.sin(P[3]), n.cos(P[3]), 0], [0, 0, 0, 1] ])
    R2 = n.matrix([ [n.cos(P[4]), 0, n.sin(P[4]), 0], [0, 1, 0, 0],
                  [-n.sin([P[4]]), 0, n.cos(P[4]), 0], [0, 0, 0, 1] ])
    R3 = n.matrix([ [n.cos(P[5]), n.sin(P[5]), 0, 0], [-n.sin(P[5]), n.cos(P[5]), 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    Z = n.matrix([ [P[6], 0, 0, 0], [0, P[7], 0, 0],
                 [0, 0, P[8], 0], [0, 0, 0, 1]])
    S = n.matrix( [ [1, P[9], P[10], 0], [0, 1, P[11], 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]] )
    return T*R1*R2*R3*Z*S


def mni2tal(pos):
    upT = spm_matrix([0, 0, 0, 0.05, 0, 0, 0.99, 0.97, 0.92])
    downT = spm_matrix([0, 0, 0, 0.05, 0, 0, 0.99, 0.97, 0.84])

    tmp = pos[-1] < 0
    pos.extend([1])
    pos = n.matrix(pos).T
    if tmp:
        pos = downT * pos
    else:
        pos = upT * pos
    return list(n.array(pos.T)[0][0:3])

def rm_info(chanPhy, rmtp):#rm_lst, rm_col=None):

    if type(rmtp) is list: rmtp = (rmtp,[chanPhy.columns[0] ])
    rm_lst, rm_col = rmtp[0], rmtp[1]

    if len(rm_col) != len(rm_lst): rm_col = [rm_col[0]]*len(rm_lst)

    for k in range(0,len(rm_lst)):
        rm_index = n.arange(0,len(chanPhy))[ [chanPhy[ rm_col[k] ].values == rm_lst[k] ] ]
        chanPhy.drop(rm_index, inplace=True, axis=0)
        chanPhy=chanPhy.set_index([list(n.arange(chanPhy.shape[0]))])

    return chanPhy

def physort(chanPhy, sortby='Brodmann'):

    if sortby == 'Brodmann':
        chanPhy, _ = MeltingSort(chanPhy, to_replace=None, value=None, colname='Brodmann')
    else: chanPhy.sort(sortby,inplace=True)

    return chanPhy

def keep_only(chanPhy,keep):

    if type(keep) is list: keep = (keep,[chanPhy.columns[0] ])
    keepLst, keepcond = keep[0], keep[1]
    if len(keepcond) != len(keepLst): keepcond = [keepcond[0]]*len(keepLst)

    keepPd = DataFrame()
    for k in range(0,len(keepLst)):
        keepPd = keepPd.append( chanPhy[chanPhy[ keepcond[k] ] == keepLst[k]] )

    return keepPd.set_index([list(n.arange(keepPd.shape[0]))])