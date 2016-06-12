import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2

def circ_corrcc(alpha, x):
    """Correlation coefficient between one circular and one linear random
    variable.
    
    Args:
        alpha: vector
            Sample of angles in radians

        x: vector
            Sample of linear random variable

    Returns:
        rho: float
            Correlation coefficient

        pval: float
            p-value

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    if len(alpha) is not len(x):
        raise ValueError('The length of alpha and x must be the same')
    n = len(alpha)

    # Compute correlation coefficent for sin and cos independently
    rxs = pearsonr(x,np.sin(alpha))[0]
    rxc = pearsonr(x,np.cos(alpha))[0]
    rcs = pearsonr(np.sin(alpha),np.cos(alpha))[0]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2));

    # Compute pvalue
    pval = 1 - chi2.cdf(n*rho**2,2);
    
    return rho, pval
