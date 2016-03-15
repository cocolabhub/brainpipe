import scipy.io as sio
import numpy

__all__ = [
    'seeg_features', 'seeg_data'
]

####################################################################
# - Features :
####################################################################
# - Load multiple features and organized them :
def seeg_features(sujet, main, Id, event, band, winL='500', rec='250', elec=None, concat_axis=0, time_idx=None):
    nb_feat = len(band)
    feat, y, channel = seeg_one_feature(sujet, main, Id[0], event, band[0], winL=winL, rec=rec)
    n_epoch, n_elec, n_pts = feat.shape
    if nb_feat > 1:
        for k in range(1, nb_feat):
            feat_one, y, channel = seeg_one_feature(sujet, main, Id[k], event, band[k], winL=winL, rec=rec)
            feat = numpy.concatenate((feat, feat_one), concat_axis)

    # -------------------------------------------------------
    # - Select an electrode and/or a window :
    if time_idx is not None:
        feat = feat[:, :, time_idx]
    if elec is not None:  # Electrode
        feat = feat[:, elec]

    return feat, numpy.ravel(y), channel[0]


# - Load one feature :
def seeg_one_feature(sujet, main, Id, event, band, winL='500', rec='250'):
    feat_path = 'C:/Users/Etienne Combrisson/Dropbox/INSERM/Classification/Features database/' + sujet + '/Features/'
    print('-> Loading ' + sujet + '-' + main + ', ' + band + ' (' + Id + ') features')

    def f(x):
        return sujet + '_' + main + '_event' + str(x) + '_' + band + '_' + winL + 'r' + rec + '_Id' + Id

    event_size = numpy.size(event)
    if event_size == 1:
        feat, channel = feat_load(feat_path, f(event))
        nb_epoch, n_elec, n_pts = feat.shape
        y = numpy.zeros((nb_epoch, 1))
    else:
        # Load the first event :
        feat_first_dim, channel = feat_load(feat_path, f(event[0]))
        # Size definition :
        nb_epoch, nb_elec, nb_pts = feat_first_dim.shape
        y = numpy.zeros((nb_epoch * event_size, 1))
        feat = numpy.zeros((nb_epoch * event_size, nb_elec, nb_pts))
        feat[0:nb_epoch, :, :] = feat_first_dim
        # Complete data :
        for k in range(1, event_size):
            feat[(k * nb_epoch):(k + 1) * nb_epoch, :, :], channel = feat_load(feat_path, f(event[k]))
            y[(k * nb_epoch):(k + 1) * nb_epoch, 0] = k

    return feat, numpy.ravel(y), channel


# - Simple function to load features :
def feat_load(feat_path, feat_name):
    mat = sio.loadmat(feat_path + feat_name)
    feat = mat['FEAT']
    channel = mat['channelb']
    return feat, channel


####################################################################
# - Load SEEG data :
####################################################################
def seeg_data(sujet, main, eventN, elec=None, win=None, path="C:/Users/Etienne Combrisson/Documents/MATLAB/Sujets/"):
    """
    Function to load SEEG data
    :param sujet: Name of the subject [Ex : 'C_rev']
    :param main: Name of the hand used [Ex : 'RH']
    :param eventN: Number of the event to load [Ex : EventN = 4]
    :param elec: Number of a specified electrode (Optionnal) [Ex : elec = 56]
    :rtype : data, channel, sf, y
    """

    # -------------------------------------------------------
    # - Path variables :
    seeg_path = path + sujet + "/Donn\xe9es/"
    seeg_templateName = 'bipolarised_' + sujet + '_mouse_' + main

    # -------------------------------------------------------
    # - Load structure's components :
    event_size = numpy.size(eventN)
    if event_size == 1:
        data, channel, sf = seeg_load(seeg_path, seeg_templateName, eventN)
        nb_elec, nb_pts, nb_epoch = data.shape
        y = numpy.zeros((nb_epoch, 1))
    else:
        # Load first element :
        data_first_dim, channel, sf = seeg_load(seeg_path, seeg_templateName, eventN[0])
        # Size definition :
        nb_elec, nb_pts, nb_epoch = data_first_dim.shape
        y = [0] * nb_epoch
        data = numpy.zeros((nb_elec, nb_pts, nb_epoch * event_size))
        data[:, :, 0:nb_epoch] = data_first_dim
        # Complete data :
        for k in range(1, event_size):
            data[:, :, (k * nb_epoch):(k + 1) * nb_epoch], channel, sf = seeg_load(seeg_path, seeg_templateName,
                                                                                   eventN[k])
            y.extend([k] * nb_epoch)  # y[(k * nb_epoch):(k + 1) * nb_epoch, 0] = k

    # -------------------------------------------------------
    # - Select an electrode and/or a window :
    if elec is not None:  # Electrode
        if win == None:  # No window
            return data[elec, :, :], channel, sf, numpy.ravel(y)
        else:  # Window
            return data[elec, (win[0]):(win[1]), :], channel, sf, numpy.ravel(y)

    else:  # No electrode
        if win == None:  # No window
            return data, channel, sf, numpy.ravel(y)
        else:  # Window
            return data[:, (win[0]):(win[1]), :], channel[0], sf, numpy.ravel(y)


# - Simple function to load data :
def seeg_load(seeg_path, seeg_templateName, event):
    seeg_name = seeg_templateName + '_' + str(event) + '.mat'
    print('Loading : ' + seeg_name)
    mat = sio.loadmat(seeg_path + seeg_name)
    data = mat['datab']
    channel = mat['channelb']
    sf = mat['sfb']

    # - Return data :
    return data, channel, int(sf)
