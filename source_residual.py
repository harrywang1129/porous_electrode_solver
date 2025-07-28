#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:29:49 2025

@author: yuhe wang
"""

import numpy as np

def source_residual(Nx, Ny, phi1, phi2, constrained_cells_1=None, lambda_vec=None, constrained_cells_2=None):
    """
    Computes the residual of the nonlinear reaction source term and applies 
    constraints using Lagrange multipliers.

    Parameters:
        phi1 (numpy.ndarray): Electrode potential vector of size (Nx*Ny,).
        phi2 (numpy.ndarray): Electrolyte potential vector of size (Nx*Ny,).
        constrained_cells_1 (list or numpy.ndarray, optional): Indices of constrained electrode cells.
        lambda_val (numpy.ndarray, optional): Lagrange multipliers.
        constrained_cells_2 (list or numpy.ndarray, optional): Indices of constrained electrolyte cells.

    Returns:
        R1 (numpy.ndarray): Residual for the electrode.
        R2 (numpy.ndarray): Residual for the electrolyte.
    """
    
    if constrained_cells_1 is None and constrained_cells_2 is None and lambda_vec is None:
        constrained = 0  # No constraints
    elif constrained_cells_1 is not None and lambda_vec is not None and constrained_cells_2 is None:
        constrained = 1  # Enforce reference in trode only
    elif constrained_cells_1 is not None and constrained_cells_2 is not None and lambda_vec is not None:
        constrained = 2  # Enforce reference in both trode & lyte
    else:
        raise ValueError("Invalid number of inputs or incompatible arguments provided.")
 
    As = 1.64e4    # specific surface area (1/m)
    E = -0.255     # reference negative electrode potential (V)
    F = 96485      # Faraday's constant (C/mol)
    R = 8.314      # Universal gas constant (J/(mol*K))
    T = 298.15     # Temperature (K)
    k = 1.7e-7     # standard reaction rate constant (m/s)
    c2 = 27.0      # v2+ concentration (mol/m³)
    c3 = 1053.0    # v3+ concentration (mol/m³)

    j0 = (F * k) * (c3**0.5) * (c2**0.5)
    E_eq = E + (R * T / F) * np.log(c3 / c2)

    a = As * j0
    b = 0.5 * F / (R * T)
    a1 = 2 * a
    a2 = -2 * a

    N = Nx * Ny
    
    # **Define Source Terms Using Nonlinear Functions**
    def f1(eta):
        return a1 * np.sinh(b * eta)
    
    def f2(eta):
        return a2 * np.sinh(b * eta)
    
    R1 = np.zeros((N,1))
    R2 = np.zeros((N,1))
    
    for i in range(N):
        eta = phi1[i] - phi2[i] - E_eq
        eta = min(eta, 0)
        R1[i] = -f1(eta)
        R2[i] = -f2(eta)
              
    # **Handle Constraints via Lagrange Multipliers**
    if constrained == 1:  # Apply constraints on phi1 (trode only)
        for i in range(len(constrained_cells_1)):
            loc = constrained_cells_1[i]
            R1[loc] += lambda_vec[i]
            
    
    elif constrained == 2:  # Apply constraints on phi1 and phi2 (trode & lyte)
        for i in range(len(constrained_cells_1)):
            loc = constrained_cells_1[i]
            R1[loc] += lambda_vec[i]
    
        for i in range(len(constrained_cells_2)):
            loc = constrained_cells_2[i]
            R2[loc - N] += lambda_vec[i + len(constrained_cells_1)]  
            
    return R1, R2

