Neuronal Features
=================


.. todo:: Missing methods in PLV // Entropy

Here's the list of implemented features:

* :ref:`sigfilt`
* :ref:`amplitude` 
* :ref:`power`
* :ref:`tf`
* :ref:`phase`
* :ref:`entropy`
* :ref:`pac`
* :ref:`pp`
* :ref:`plp`
* :ref:`plv`



.. code-block:: python

    from brainpipe.features import *

Basics
------

.. _sigfilt:

Filtered signal
~~~~~~~~~~~~~~~
.. automodule:: feature
   :members: sigfilt
   :noindex:

.. _amplitude:

Amplitude
~~~~~~~~~
.. automodule:: feature
   :members: amplitude
   :noindex:

.. _power:

Power
~~~~~
.. automodule:: feature
   :members: power
   :noindex:

.. _tf:

Time-Frequency
~~~~~~~~~~~~~~
.. automodule:: feature
   :members: TF
   :noindex:

.. figure::  ../images/tf.png
   :align:   center

.. _phase:

Phase
~~~~~
.. automodule:: feature
   :members: phase
   :noindex:

.. _entropy:

Entropy
~~~~~~~

Coupling features
-----------------

.. _pac:

Phase-Amplitude Coupling
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: feature
   :members: pac
   :noindex:

.. figure::  ../images/pac.png
   :align:   center

.. _pp:

Prefered-phase
~~~~~~~~~~~~~~~~~~~

.. _plp:

Phase-locked power
~~~~~~~~~~~~~~~~~~~
.. automodule:: feature
   :members: PhaseLockedPower
   :noindex:
.. figure::  ../images/plp.png
   :align:   center

.. _plv:


Phase-Locking Value
~~~~~~~~~~~~~~~~~~~

Tools
------
.. autofunction:: cfcVec

.. autofunction:: cfcRndSignals




