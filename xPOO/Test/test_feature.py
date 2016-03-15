import pickle
from os.path import dirname
import brainpipe
from brainpipe.xPOO.feature import power


def test_load_dataset():
    """Load the 3chan-Dataset"""

    with open(dirname(brainpipe.__file__)+'/xPOO/Test/3chan-Dataset.pickle', "rb") as f:
        data = pickle.load(f)
    return data['x'], data['y'], data['channel'], data['sf']


def test_power_in_band():
    """Compute the power in three specific bands"""

    # Load dataset:
    x, y, channel, sf = test_load_dataset()

    # Define the frequency for the power :
    f = [[2, 4], [8, 13], [60, 200]]

    # Define power propreties:
    poObj = power(sf, f, x.shape[1],
                  method='wavelet',
                  baseline=(250, 750),
                  norm=3,
                  step=50,
                  width=100,
                  split=(None, None, 20)
                  )

    # Compute power :
    po = poObj.get(x)

    # Plot power :
    poObj.plot(po, linewidth=2)


def test_timefrequency_map():
    """Compute the power in three specific bands"""

    # Load dataset:
    x, y, channel, sf = test_load_dataset()

    # Define power propreties:
    poObj = power(sf, [2, 200], x.shape[1],
                  method='wavelet',
                  baseline=(250, 750),
                  norm=3,
                  step=50,
                  width=100,
                  split=(None, None, 20)
                  )

    # Compute power :
    po = poObj.tf(x, f=(2, 200, 20, 10))

    # Plot power :
    poObj.tfplot(po, interpolation='none', interp=(0.1, 1))


def test_phase_in_band():
    """Compute the phase in two specific bands"""

    # Load dataset:
    x, y, channel, sf = test_load_dataset()

    # Define the frequency for the power :
    f = [[0.1, 1.5], [2, 4]]

    # Define power propreties:
    phaObj = phase(sf, f, x.shape[1],
                   step=50,
                   width=100,
                   )

    # Compute power :
    pa = phaObj.get(x)

    # Plot power :
    phaObj.plot(pa, linewidth=2)
