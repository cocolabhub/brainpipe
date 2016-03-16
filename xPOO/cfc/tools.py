import numpy as n

__all__ = ['rndCfcSignals']


def rndCfcSignals(fPha=2, fAmp=100, sf=1024, ndatasets=10,
                  tmax=1, chi=0, noise=1, dPha=0, dAmp=0):
    """Generate randomly phase-amplitude coupled signals.

    Parameters
    ----------
    fPha : int/float, optional, [def : 2]
        Frequency for phase

    fAmp : int/float, optional, [def : 100]
        Frequency for amplitude

    sf : int, optional, [def : 1024]
        Sampling frequency

    ndatasets : int, optional, [def : 10]
        Number of datasets

    tmax : int/float (1<=tmax<=3), optional, [def : 1]
        Length of the time vector. If tmax=2 and sf=1024,
        the number of time points npts=1024*2=2048

    chi : int/float (0<=chi<=1), optional, [def : 0]
        Amout of coupling. If chi=0, signals of phase and amplitude
        are strongly coupled.

    noise : int/float (1<=noise<=3), optional, [def : 1]
        Amount of noise

    dPha : int/float (0<=dPha<=100), optional, [def : 0]
        Introduce a random incertitude on the phase frequency.
        If fPha is 2, and dPha is 50, the frequency for the phase signal
        will be between :
        [2-0.5*2, 2+0.5*2]=[1,3]

    dAmp : int/float (0<=dAmp<=100), optional, [def : 0]
        Introduce a random incertitude on the amplitude frequency.
        If fAmp is 60, and dAmp is 10, the frequency for the amplitude
        signal will be between :
        [60-0.1*60, 60+0.1*60]=[54,66]

    Return
    ----------
    data : array
        The randomly coupled signals. The shape of data will be
        (ndatasets x npts)

    time : array
        The corresponding time vector
    """
    # Check the inputs variables :
    if (tmax < 1) or (tmax > 3):
        tmax = 1
    if (chi < 0) or (chi > 1):
        chi = 0
    if (noise < 1) or (noise > 3):
        noise = 1
    if (dPha < 0) or (dPha > 100):
        dPha = 0
    if (dAmp < 0) or (dAmp > 100):
        dAmp = 0
    fPha, fAmp = n.array(fPha), n.array(fAmp)
    time = n.arange(0, tmax, 1/sf)

    # Delta parameters :
    aPha = [fPha*(1-dPha/100), fPha*(1+dPha/100)]
    deltaPha = aPha[0] + (aPha[1]-aPha[0])*n.random.rand(ndatasets, 1)
    aAmp = [fAmp*(1-dAmp/100), fAmp*(1+dAmp/100)]
    deltaAmp = aAmp[0] + (aAmp[1]-aAmp[0])*n.random.rand(ndatasets, 1)

    # Generate the rnd datasets :
    data = n.zeros((ndatasets, len(time)))
    for k in range(ndatasets):
        # Create signals :
        xl = n.sin(2*n.pi*deltaPha[k]*time)
        xh = n.sin(2*n.pi*deltaAmp[k]*time)
        e = noise*n.random.rand(len(xl))

        # Create the coupling :
        ah = 0.5*((1 - chi) * xl + 1 + chi)
        al = 1
        data[k, :] = (ah*xh) + (al*xl) + e

    return data, time
