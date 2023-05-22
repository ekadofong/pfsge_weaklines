import numpy as np
import extinction

# The following comes close to but does not exactly match the Valentino 2017
# (http://adsabs.harvard.edu/abs/2017MNRAS.472.4878V) predictions,
# probably because of use of slightly different extinction laws.
# 

f = 0.76 # stellar to nebular Av ratio. # [EKF: does this get used??]

def charlot_and_fall_extinction(lam,dust1,dust2,dust2_index,dust1_index=-1.0,kriek=True):
   """
   returns F(obs) / F(emitted) for a given attenuation curve (dust_index) + dust1 + dust2
   """

   dust1_ext = np.exp(-dust1*(lam/5500.)**dust1_index)
   dust2_ext = np.exp(-dust2*(lam/5500.)**dust2_index)

   # sanitize inputs
   lam = np.atleast_1d(lam).astype(float)

   # are we using Kriek & Conroy 13?
   if kriek:
      dd63 = 6300.00
      lamv = 5500.0
      dlam = 350.0
      lamuvb = 2175.0

      #Calzetti curve, below 6300 Angstroms, else no addition
      cal00 = np.zeros_like(lam)
      gt_dd63 = lam > dd63
      le_dd63 = ~gt_dd63
      if np.sum(gt_dd63) > 0:
         cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
      if np.sum(le_dd63) > 0:
         cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-0.198*(1E4/lam[le_dd63])**2 + 0.011*(1E4/lam[le_dd63])**3) + 1.78
      cal00 = cal00/0.44/4.05 

      eb = 0.85 - 1.9 * dust2_index  #KC13 Eqn 3

      #Drude profile for 2175A bump
      drude = eb*(lam*dlam)**2 / ( (lam**2-lamuvb**2)**2 + (lam*dlam)**2 )

      attn_curve = dust2*(cal00+drude/4.05)*(lam/lamv)**dust2_index
      dust2_ext = np.exp(-attn_curve)

   ext_tot = dust2_ext*dust1_ext

   return ext_tot

def predict_L_Ha(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the Halpha luminosity given logSFR/(Msun/yr) and stellar Av.
   Reproduces Valentino 2017 catalog a bias of 0.005 dex
   and standard deviation of 0.007 dex.
   return: log of the Halpha luminosity in erg/s
   """
      
   logL_Ha_int = logSFR - np.log10(7.9e-42) # K98
   ext = charlot_and_fall_extinction(np.array([6564.61]), dust1, dust2, n)
   
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.   
   logL_Ha = logL_Ha_int + extinction_factor

   return logL_Ha
   
def predict_L_Hb(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the Hbeta luminosity given logSFR/(Msun/yr) and stellar Av.
   Reproduces Valentino 2017 catalog a bias of 0.024 dex
   and standard deviation of 0.026 dex.
   return: log of the Hbeta luminosity in erg/s
   """
   
   logL_Hb_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) # K98 and Case B
   ext = charlot_and_fall_extinction(np.array([4862.68]), dust1, dust2, n)
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.     
   logL_Hb = logL_Hb_int + extinction_factor
   
   return logL_Hb
   
   
def predict_L_Hg(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the Hgamma luminosity given logSFR/(Msun/yr) and stellar Av.
   return: log of the Hgamma luminosity in erg/s
   """
   
   # K98 and Case B
   logL_Hg_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.466)
   ext = charlot_and_fall_extinction(np.array([4341.68]), dust1, dust2, n)
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.        
   logL_Hg = logL_Hg_int + extinction_factor
   
   return logL_Hg
   

def predict_L_Hd(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the Hdelta luminosity given logSFR/(Msun/yr) and stellar Av.
   return: log of the Hdelta luminosity in erg/s
   """
   
   # K98 and Case B
   logL_Hd_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.256)
   ext = charlot_and_fall_extinction(np.array([4102.89]), dust1, dust2, n)
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.           
   logL_Hd = logL_Hd_int + extinction_factor
   
   return logL_Hd
   

def predict_L_He(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the Hepsilon luminosity given logSFR/(Msun/yr) and stellar Av.
   return: log of the Hdelta luminosity in erg/s
   """
   
   # K98 and Case B
   logL_He_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.158)
   ext = charlot_and_fall_extinction(np.array([3971.19]), dust1, dust2, n)         
   logL_He = logL_He_int + extinction_factor
   
   return logL_He
   
def predict_L_Hz(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
   return: log of the Hdelta luminosity in erg/s
   """
   
   # K98 and Case B
   logL_Hz_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.105)
   ext = charlot_and_fall_extinction(np.array([3890.17]), dust1, dust2, n)
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.              
   logL_Hz = logL_Hz_int + extinction_factor
   
   return logL_Hz
   
   
def predict_L_OII_tot(logSFR, n, dust1, dust2, include_extinction=False):
   """
   Predict the [O II] luminosity given logSFR/(Msun/yr) and stellar Av.
   Reproduces Valentino 2017 catalog a bias of 0.058 dex
   and standard deviation of 0.043 dex.
   return: log of the [O II] luminosity in erg/s
   """
   
   logL_OII_int = logSFR - np.log10(7.9e-42) # K04 actually just K98
   ext = charlot_and_fall_extinction(np.array([3728.0]), dust1, dust2, n) 
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.                
   logL_OII = logL_OII_int + extinction_factor
   
   return logL_OII
   
   
def predict_L_OIII5007(logSFR, n, dust1, dust2, logMstar, include_extinction=False):
   """
   Predict the [O III] luminosity given logSFR/(Msun/yr) and stellar Av.
   Reproduces Valentino 2017 catalog a bias of 0.025 dex
   and standard deviation of 0.019 dex.
   return: log of the [O III] luminosity in erg/s
   """
   
   logL_Hb = predict_L_Hb(logSFR, n, dust1, dust2, include_extinction=include_extinction)
   logOIII_Hbeta = 0.3 + 0.48*np.arctan(-(logMstar - 10.28))
   
   logL_OIII = logL_Hb + logOIII_Hbeta
   
   return logL_OIII
   
   
def predict_L_NeIII3870(logSFR, n, dust1, dust2, logMstar, include_extinction=False):
   """
   Predict the [Ne III] 3870 luminosity given logSFR/(Msun/yr) and stellar Av and stellar mass.
   return: log of the [Ne III] luminosity in erg/s
   """
   
   # Get Hbeta assuming no dust
   
   logL_Hb_int = predict_L_Hb(logSFR, 0.0, 0.0, 0.0)
   
   # Get [O III] assuming no dust
   logOIII_Hbeta = 0.3 + 0.48*np.arctan(-(logMstar - 10.28))
   logL_OIII_int = logL_Hb_int + logOIII_Hbeta
   
   # Assuming [Ne III] is 10x weaker than [O III]
   logL_NeIII_int = logL_OIII_int - 1.0
   
   # Apply dust
   ext = charlot_and_fall_extinction(np.array([3869.85]), dust1, dust2, n)  
   if include_extinction:
      extinction_factor = np.log10(ext)
   else:
      extinction_factor = 0.               
   logL_NeIII = logL_NeIII_int + extinction_factor
   
   return logL_NeIII
   
   
   
def predict_L_NII6585(logSFR, n, dust1, dust2, logMstar, include_extinction=False):
   """Best fit relation from Strom+2017"""

   logL_Ha = predict_L_Ha(logSFR, n, dust1, dust2, include_extinction=include_extinction)
   logL_Hb = predict_L_Hb(logSFR, n, dust1, dust2, include_extinction=include_extinction)
   logL_OIII = predict_L_OIII5007(logSFR, n, dust1, dust2, logMstar)
   
   logL_NII  = 0.61/((logL_OIII - logL_Hb) - 1.12) + 0.22 + logL_Ha
   
   return logL_NII
   
   
def predict_L_SII_tot(logSFR, n, dust1, dust2, logMstar, include_extinction=False):
   """Best fit relation from Strom+2017"""

   logL_Ha = predict_L_Ha(logSFR, n, dust1, dust2, include_extinction=include_extinction)
   logL_Hb = predict_L_Hb(logSFR, n, dust1, dust2, include_extinction=include_extinction)
   logL_OIII = predict_L_OIII5007(logSFR, n, dust1, dust2, logMstar)
   
   logL_SII  = 0.72/((logL_OIII - logL_Hb) - 1.15) + 0.53 + logL_Ha
   
   return logL_SII
   


