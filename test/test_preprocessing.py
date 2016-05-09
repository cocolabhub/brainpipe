from brainpipe.preprocessing import bipolarization, xyz2phy
import numpy as np


def test_bipolarization():
    data = np.random.rand(104, 9, 160)
    channel = ['l0', 'l1', 'l2', 'l3', 'r1', 'r2', 'r3', 'eog', 'emg']
    dataB, chanB, _ = bipolarization(data, channel, dim=1)
    print('-> Test bipolarization: OK')


def test_physio():
    xyz = np.array([[2.439394, -54.058218, 51.563646],
                    [6.262626, -54.058218, 51.563646],
                    [44.444444, -54.058218, 51.563646],
                    [16.161616, -23.529709, 60.792589]])
    channel = ['p2', 'p3', 'e4', 'i5']

    phyobj = xyz2phy()
    dat = phyobj.get(xyz, channel=channel)
    print('-> Test physio: OK')


if __name__ == '__main__':
    test_bipolarization()
    test_physio()
