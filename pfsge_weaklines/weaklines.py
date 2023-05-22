import numpy as np
import pyneb as pn
from . import johnson_elines

def predict_L_OIII_4363 (logSFR, n, dust1, dust2, logMstar, TOIII, ne=100., include_extinction=False):
    """
    Predicts the logarithm of the [OIII]λ4363 line luminosity based off of the line ratio between
    [OIII]4363/[OIII]5007, which is a temperature-sensitive ratio.

    Parameters:
        logSFR (float): Logarithm of the star formation rate
        n (float): power-law index of Dust parameter 2
        dust1 (float): Dust parameter 1
        dust2 (float): Dust parameter 2
        logMstar (float): Logarithm of the stellar mass
        TOIII (float): Electron temperature of OIII in K
        ne (float, optional): Electron density in cm^-3. Defaults to 100.
        include_extinction (bool, optional): Flag indicating whether to include extinction. Defaults to False.

    Returns:
        float: Logarithm of the [OIII]λ4363 line luminosity.
    """    
    log_LOIII5007_int = johnson_elines.predict_L_OIII5007 ( logSFR, n, dust1, dust2, logMstar, include_extinction=False ) 
    
    opp = pn.Atom ( 'O', 3 )
    j_OIII5007 = opp.getEmissivity ( TOIII, ne, *opp.getTransition(5007.) )
    j_OIII4363 = opp.getEmissivity ( TOIII, ne, *opp.getTransition(4363.) )
    
    log_LOIII4363_int = log_LOIII5007_int + np.log10(j_OIII4363 / j_OIII5007)
    
    ext = johnson_elines.charlot_and_fall_extinction(np.array([4363.]), dust1, dust2, n) 
    if include_extinction:
        extinction_factor = np.log10(ext)
    else:
        extinction_factor = 0.           
        
    log_LOIII4363  = log_LOIII4363_int + extinction_factor
    return log_LOIII4363

def predict_L_NeIII3969 ( logSFR, n, dust1, dust2, logMstar, include_extinction=False):
    """
    Predicts the logarithm of the [NeIII]3969 line luminosity from its fixed ratio with [NeIII]3870

    Parameters:
        logSFR (float): Logarithm of the star formation rate
        n (float): power-law index of Dust parameter 2
        dust1 (float): Dust parameter 1
        dust2 (float): Dust parameter 2
        logMstar (float): Logarithm of the stellar mass
        TOIII (float): Electron temperature of OIII in K
        ne (float, optional): Electron density in cm^-3. Defaults to 100.
        include_extinction (bool, optional): Flag indicating whether to include extinction. Defaults to False.

    Returns:
        float: Logarithm of the [OIII]λ4363 line luminosity.
    """     
    log_LNeIII3870_int = johnson_elines.predict_L_NeIII3870 ( logSFR, n, dust1, dust2, logMstar, include_extinction=False )
    
    nepp = pn.Atom('Ne',3)
    
    te = 1e4 
    ne = 1e2 # \\ these shouldn't actually matter because the LR is fixed
    j_NeIII3870 = nepp.getEmissivity ( te, ne, *nepp.getTransition(3870.) )
    j_NeIII3969 = nepp.getEmissivity ( te, ne, *nepp.getTransition(3969.) )    
    
    log_LNeIII3969_int = log_LNeIII3870_int + np.log10(j_NeIII3969/j_NeIII3870)
    
    ext = johnson_elines.charlot_and_fall_extinction(np.array([3969.]), dust1, dust2, n) 
    if include_extinction:
        extinction_factor = np.log10(ext)
    else:
        extinction_factor = 0.           
    
    log_LNeIII3969 = log_LNeIII3969_int + extinction_factor
    return log_LNeIII3969

def predict_L_CII1909 ():
    raise NotImplementedError