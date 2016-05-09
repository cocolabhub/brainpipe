Neuronal Features
=================


.. todo:: Missing methods in pac // PLV // Entropy

Here's the list of implemented features:

* :ref:`sigfilt`
* :ref:`amplitude` 
* :ref:`power`
* :ref:`tf`
* :ref:`phase`
* :ref:`entropy`
* :ref:`pac`
* :ref:`pp`
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

.. _pp:

Prefered-phase
~~~~~~~~~~~~~~~~~~~

.. _plv:

Phase-Locking Value
~~~~~~~~~~~~~~~~~~~

Tools
------
.. autofunction:: cfcVec

.. autofunction:: cfcRndSignals




