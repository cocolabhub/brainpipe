import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedShuffleSplit, KFold, permutation_test_score, GridSearchCV)
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, RFECV
from sklearn.pipeline import Pipeline, FeatureUnion


__all__ = ['MFpipe']


class MFpipe(object):

    """Define a pipeline of multi-features

    Args:
        y: ndarray
            Vector label

        cv: sklearn.cross_validation, optional, (def: default)
            An external cross-validation to split in training and testing
            to validate the pipeline without overfiting.               

        random_state: ineteger, optional, (def: 0)
            Fix the random state of the machine for reproducibility.

    Return
        Multi-features pipeline object
    """

    def __init__(self, y, cv, random_state=0):
        # Get variables :
        self._cv = cv
        self._nfolds = len(self._cv)
        self._y = y
        self._nclass = len(np.unique(y))

        # Check cross-validation :
        if not self._cv.random_state:
            self._cv.random_state = random_state

        
    def fit(self, x, rep=1, n_iter=5, n_jobs=1, verbose=0):
        """Apply the pipeline.

        Args:
            x: ndarray
                Array of features organize as (n_trials, n_features)

        Kargs:
            rep: integer, optional, (def: 1)
                Number of repetitions for the whole pipeline.

            n_iter: integer, optional, (def: 5)
                Number of iterations in order to find the best set of
                parameters

            n_jobs: integer, optional, (def: 1)
                Number of jobs for parallel computing. Use -1 for all jobs.

            verbose: integer, optional, (def: 0)
                Control displaying state 

        Return
            da: the final vector of decoding accuracy of shape (n_repetitions,)
        """
    
        # Pre-defined matrix :
        da = np.zeros((rep,), dtype=np.float32)
        self.y_predict, self.y_test = [], []
        self.best_estimator_, self.best_params_ = [], []
        outstr = 'REP: {r}/'+str(rep)+', FOLD: {f}/'+str(self._nfolds)+' = {param}'

        # Loop on repetitions :
        for r in range(rep):

            # Pre-defined label vectors and estimator/params:
            yAll_pred, yAll_true = [], []
            best_estimator_, best_params_ = [], []

            # Loop on training/testing index :
            for f, (trainIdx, testIdx) in enumerate(self._cv):

                # Training/testing set :
                xtrain, ytrain = x[trainIdx, :], self._y[trainIdx]
                xtest, ytest = x[testIdx, :], self._y[testIdx]

                # Cross-validate the choice of parameters  :
                cv = StratifiedShuffleSplit(ytrain, n_iter=n_iter, test_size=0.2)
                grid = GridSearchCV(self.pipeline, cv=cv, param_grid=self.grid, verbose=verbose, n_jobs=n_jobs)
                grid.fit(xtrain, ytrain)
                grid.best_estimator_.fit(xtrain, ytrain)

                # Save best estimator/params for each fold :
                best_estimator_.append(grid.best_estimator_)
                best_params_.append(grid.best_params_)

                # Step print :
                if verbose > 0:
                    print(outstr.format(r=r+1, f=f+1, param=str(grid.best_params_)))

                # Evaluate on testing :
                yAll_pred.extend(grid.predict(xtest))
                yAll_true.extend(ytest)

            # Save best estimator/params for each repetition :
            self.best_estimator_.append(best_estimator_)
            self.best_params_.append(best_params_)
            self.y_predict.append(yAll_pred)
            self.y_test.append(yAll_true)

            # Evaluate the decoding of this repetition :
            da[r] =  accuracy_score(yAll_true, yAll_pred)*100

            # Update random_state for the next repetition :
            self._cv.random_state += 1

                # score, daperm, pvalue = permutation_test_score(grid.best_estimator_, xtest, ytest, scoring="accuracy", cv=3,
                #                                                n_permutations=100, n_jobs=1)
                # print('SCORE: ', score, 'DAPERM: ', daperm, 'PVALUE: ', pvalue)

        return da


    def custom_pipeline(self, pipeline=None, grid=None):
        """Send a custom pipeline

        Kargs:
            pipeline: sklearn.Pipeline, optional, (def: None)
                The pipeline to use

            grid: sklearn.GridCv, optional, (def: None)
                A grid for parameters optimisation
        """
        self.pipeline, self.grid = pipeline, grid


    def default_pipeline(self, name, n_pca=10, n_best=10, lda_shrink=10, svm_C=10,
                         svm_gamma=10, fdr_alpha=[0.05], fpr_alpha=[0.05]):
        """Use a default combination of parameters for building a pipeline

        Args:
            name: string
                The string for building a default pipeline (see examples below)

        Kargs:
            n_pca: integer, optional, (def: 10)
                The number of components to search

            n_best: integer, optional, (def: 10)
                Number of best features to consider using a statistical method

            lda_shrink: integer, optional, (def: 10)
                Fit optimisation parameter for the lda

            svm_C/svm_gamma: integer, optional, (def: 10/10)
                Parameters to optimize for the svm

            fdr/fpr_alpha: list, optional, (def: [0.05])
                List of float for selecting features using a fdr or fpr

        Examples:
            >>> # Basic classifiers :
            >>> name = 'lda' # or name = 'svm_linear' for a linear SVM
            >>> # Combine a classifier with a feature selection method :
            >>> name = 'lda_fdr_fpr_kbest_pca'
            >>> # The method above will use an LDA for the features evaluation
            >>> # and will combine a FDR, FPR, k-Best and pca feature seelction.
            >>> # Now we can combine with classifier optimisation :
            >>> name = 'lda_optimized_pca' # will try to optimize an LDA with a pca
            >>> name = 'svm_kernel_C_gamma_kbest' # optimize a SVM by trying
            >>> # diffrent kernels (linear/RBF), and optimize C and gamma parameters
            >>> # combine with a k-Best features selection.
        """
        # ----------------------------------------------------------------
        # DEFINED COMBINORS
        # ----------------------------------------------------------------
        self._piperef = ['lda_pca', 'lda_pca-kBest', 'lda_optimized_pca', 'lda_optimized_pca-kBest']
        pca = PCA()
        selection = SelectKBest()
        scaler = StandardScaler()
        fdr = SelectFdr()

        # ----------------------------------------------------------------
        # RANGE DEFINITION
        # ---------------------------------------------------------
        pca_range = np.arange(1, n_pca+1)
        kbest_range = np.arange(1, n_best+1)
        C_range = np.logspace(-5, 15, svm_C, base=2.) #np.logspace(-2, 2, svm_C)
        gamma_range = np.logspace(-15, 3, svm_gamma, base=2.) #np.logspace(-9, 2, svm_gamma)

        # Check range :
        if not kbest_range.size: kbest_range = [1]
        if not pca_range.size: pca_range = [1]
        if not C_range.size: C_range = [1.]
        if not gamma_range.size: gamma_range = ['auto']

        # ----------------------------------------------------------------
        # DEFINED PIPELINE ELEMENTS
        # ----------------------------------------------------------------
        pipeline = []
        grid = {}
        combine = []

        # ----------------------------------------------------------------
        # BUILD CLASSIFIER
        # ----------------------------------------------------------------
        # -> SCALE :
        if name.lower().find('scale') != -1:
            pipeline.append(("scaler", scaler))

        # -> LDA :
        if name.lower().find('lda') != -1:

            # Default :
            if name.lower().find('optimized') == -1:
                clf = LinearDiscriminantAnalysis(priors=np.array([1/self._nclass]*self._nclass))

            # Optimized :
            elif name.lower().find('optimized') != -1:
                clf = LinearDiscriminantAnalysis(priors=np.array([1/self._nclass]*self._nclass), solver='lsqr')
                grid['clf__shrinkage'] = np.linspace(0., 1., lda_shrink)

        # -> SVM :
        elif name.lower().find('svm') != -1:

            # Linear/RBF standard kernel :
            if name.lower().find('linear') != -1:
                kwargs = {'kernel':'linear'}
            elif name.lower().find('rbf') != -1:
                kwargs = {'kernel':'rbf'}
            else:
                kwargs = {}

            # Optimized :
            if name.lower().find('optimized') != -1:

                # Kernel optimization :
                if name.lower().find('kernel') != -1:
                    grid['clf__kernel'] = ('linear', 'rbf')

                # C optimization :
                if name.lower().find('_c_') != -1:
                    grid['clf__C'] = C_range

                # Gamma optimization :
                if name.lower().find('gamma') != -1:
                    grid['clf__gamma'] = gamma_range


            clf = SVC(**kwargs)

        # ----------------------------------------------------------------
        # BUILD COMBINE
        # ----------------------------------------------------------------
        # -> FDR :
        if name.lower().find('fdr') != -1:
            combine.append(("fdr", fdr))
            grid['features__fdr__alpha'] = fdr_alpha

        # -> FPR :
        if name.lower().find('fpr') != -1:
            combine.append(("fpr", fdr))
            grid['features__fpr__alpha'] = fpr_alpha        

        # -> PCA :
        if name.lower().find('pca') != -1:
            combine.append(("pca", pca))
            grid['features__pca__n_components'] = pca_range

        # -> kBest :
        if name.lower().find('kbest') != -1:
            combine.append(("kBest", selection))
            grid['features__kBest__k'] = kbest_range

        # -> RFECV :
        if name.lower().find('rfecv') != -1:
            rfecv = RFECV(clf)
            combine.append(("RFECV", rfecv))

        self.combine = FeatureUnion(combine)

        # ----------------------------------------------------------------
        # SAVE PIPELINE
        # ----------------------------------------------------------------
        # Build ordered pipeline :
        if len(combine):
            pipeline.append(("features", self.combine))
        pipeline.append(("clf", clf))

        # Save pipeline :
        self.pipeline = Pipeline(pipeline)
        self.grid = grid

        # print('\nCOMBINE: ', self.combine, '\n\nPIPELINE: ', self.pipeline, '\n\nGRID: ', self.grid)
