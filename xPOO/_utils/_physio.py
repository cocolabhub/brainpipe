import numpy as n


def pos2label(pos, mask, hdr, gray, label, r=5, nearest=True):
    """From the xyz coordonates, get the labels of gyrus, hemisphere...
    """
    pos = mni2tal(pos)
    ind = coord2ind(pos, mask, hdr, gray, r=r, nearest=nearest)
    # Not found
    if ind == -1:
        hemi, lobe, gyrus, matter, brod = ['Not found']*4+[-1]
    # Not found <r:
    elif ind == -2:
        hemi, lobe, gyrus, matter, brod = [
            'No Gray Matter found within +/-'+str(r)+'mm']*4+[0]
    # found:
    else:
        hemi, lobe, gyrus, matter, brod = [label[ind, :][0][0]], [label[ind, :][1][0]], [
            label[ind, :][2][0]], [label[ind, :][3][0]], [label[ind, :][4][0]]
    return hemi, lobe, gyrus, matter, brod


def coord2ind(pos, mask, hdr, gray, r=5, nearest=True):
    """Find the correspondance between coordonate and atlas index
    """
    # Apply a transformation of position:
    pos.extend([1])
    sub = list(
        n.around(n.array(n.linalg.lstsq(hdr, n.matrix(pos).T)[0].T)[0]).astype(
            int))
    # Find the index with the nearest option:
    if nearest:
        if sub[2] > gray.shape[2]:
            sub[2] = gray.shape[2]-1
        tranche = n.squeeze(gray[:, :, sub[2]])
        # Euclidian distance :
        dist = 100*n.ones((tranche.shape))
        u, v = n.where(tranche == 1)
        for k in range(0, len(u)):
            dist[u[k], v[k]] = n.math.sqrt((sub[1]-v[k])**2 + (sub[0]-u[k])**2)
        mindist = dist.min()
        umin, vmin = n.where(dist == mindist)
        if mindist < r:
            ind = mask[umin[0], vmin[0], sub[2]]-1
        else:
            ind = -2
    else:
        try:
            ind = mask[sub[0], sub[1], sub[2]]-1
        except:
            ind = -1
    return ind


def loadatlas(atlas='tal'):
    """Load the atlas from the brainpipe module
    """
    B3Dpath = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    # Load AAL atlas :
    if atlas is 'AAL':  # PAS FINIT
        AAL = scio.loadmat(B3Dpath + '/Atlas/Labels/BrainNet_AAL_Label')
        hdr, label, mask = AAL['AAL_hdr'], AAL['AAL_label'], AAL['AAL_vol']

    # Load talairach atlas :
    if atlas is 'tal':
        TAL = scio.loadmat(B3Dpath + '/Atlas/Labels/Talairach_atlas')
        hdr = TAL['hdr']['mat'][0][0]

        label, mask, gray = TAL['label'], TAL['mask'], TAL['gray']
        brodtxt, brodidx = TAL['brod']['txt'][0][0], TAL['brod']['idx'][0][0]

    return hdr, mask, gray, brodtxt, brodidx, label


def mni2tal(pos):
    """Transform coordonates from mni to talairach
    """
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


def spm_matrix(P):
    """Matrix transformation
    """

    q = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    P.extend(q[len(P):12])

    T = n.matrix(
        [[1, 0, 0, P[0]], [0, 1, 0, P[1]], [0, 0, 1, P[2]], [0, 0, 0, 1]])
    R1 = n.matrix([[1, 0, 0, 0], [0, n.cos(P[3]), n.sin(P[3]), 0],
                   [0, -n.sin(P[3]), n.cos(P[3]), 0], [0, 0, 0, 1]])
    R2 = n.matrix([[n.cos(P[4]), 0, n.sin(P[4]), 0], [0, 1, 0, 0],
                   [-n.sin([P[4]]), 0, n.cos(P[4]), 0], [0, 0, 0, 1]])
    R3 = n.matrix([[n.cos(P[5]), n.sin(P[5]), 0, 0], [-n.sin(P[5]), n.cos(
                   P[5]), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    Z = n.matrix([[P[6], 0, 0, 0], [0, P[7], 0, 0],
                  [0, 0, P[8], 0], [0, 0, 0, 1]])
    S = n.matrix([[1, P[9], P[10], 0], [0, 1, P[11], 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    return T*R1*R2*R3*Z*S
