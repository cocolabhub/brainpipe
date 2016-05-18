Brainpipe
*********

Brainpipe is a python toolbox dedicated for neuronal signals analysis and machine-learning. The aim is to provide a variety of tools to extract informations from neural activities (features) and use machine-learning to validate hypothesis. The machine-learning used the excellent `scikit-learn <http://scikit-learn.org/stable/>`_  library. Brainpipe can also perform parallel computing and try to optimize RAM usage for our 'large' datasets. 

It's evolving every day! So if you have problems, bugs or if you want to collaborate and add your own tools, contact me at e.combrisson@gmail.com

.. figure::  ../images/titend.png
   :align:   center

   (My amazing `Jhenn Oz <https://www.facebook.com/jhenntattooist/?fref=ts>`_)

	

Requirement
***********
brainpipe is developed on Python 3, so the compatibility with python 2 is not guaranted! (not tested yet)

Please, check if you have this toolbox installed and already up-to-date:

- matplotlib (visualization)
- scikit-learn (machine learning)
- joblib (parallel computing)

Installation
************
For instance, the easiest way of installing brainpipe is to ues github (`brainpipe <https://github.com/EtienneCmb/brainpipe>`_ ). 

Go to your python site-package folder (ex: anaconda3/lib/python3.5/site-packages) and in a terminal run

.. code-block:: python

	git clone git@github.com:EtienneCmb/brainpipe.git

What's new
**********
v0.1.0
=======

- Statistics:
	- Permutations: array optimized permutation module
	- p-values on permutations can be compute on 1 tail (upper and lower part) or two tails
	- metric: 
	- Multiple comparison: maximum statistique, Bonferroni, FDR
- Features:
	- sigfilt//amplitude//power//TF: wilcoxon/Kruskal-Wallis/permutations stat test (comparison with baseline)
	- PAC: new Normalized Direct PAC method (Ozkurt, 2012)
- Visualization:
	- tilerplot() with plot1D() and plot2D() with automatic control of subplot
- Tools:
	- Array: ndsplit and ndjoin method which doesn't depend on odd/even size (but return list)
	- squarefreq() generate a square frequency vector
- Bugs:
	- Probably on stat for classification() and multi-features, but not tested. It might works
	

Organization
************
.. toctree::
   :maxdepth: 3
   :caption: PROCESSING

   preprocessing
   feature

.. toctree::
   :maxdepth: 3
   :caption: CLASSIFICATION

   classification

.. toctree::
   :maxdepth: 3
   :caption: STATISTICS

   stat

.. toctree::
   :maxdepth: 3
   :caption: VISUALIZATION

   visualization

.. toctree::
   :maxdepth: 3
   :caption: OTHERS

   tools



Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

