#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:41:46 2025

@author: yuhe wang
"""
#!/usr/bin/env python3
import math
import numpy as np

def bultervolmerclassic(phi_e, phi_l, ca, cc):
    """
    Computes the classical Butler–Volmer equation.
    
    Parameters:
      phi_e, phi_l: Electrode and electrolyte potentials (scalar).
      ca, cc: Anion and cation concentrations (scalar).
      
    Returns:
      aj: Local current density (A/m²)
      j: Interfacial current density (A/m²)
      eta: Overpotential (V)
    """
    F = 96485.0     # C/mol
    R = 8.314       # J/(mol*K)
    T = 298.15      # K
    k = 1.7e-7      # m/s
    E0 = -0.255     # V
    As = 1.64e4     # 1/m

    frt = F / (R * T)
    j0 = (F * k) * (ca ** 0.5) * (cc ** 0.5)
    E = E0 + (1.0 / frt) * math.log(ca / cc)
    eta = phi_e - phi_l - E
    eta = np.where(eta > 0, 0, eta)
    frt05 = 0.5 * frt
    e1 = np.exp(frt05 * eta)    
    e2 = np.exp(-frt05 * eta) 
    j = j0 * (e1 - e2)
    aj = As * j
    return aj, j, eta


