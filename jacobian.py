#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:16:30 2025

@author: yuhe wang
"""

#!/usr/bin/env python3
import numpy as np


def jacobian(Nx, Ny, A1, A2, phi1, phi2, constrained_cells_1=None, constrained_cells_2=None):
    """
    Constructs the Jacobian matrix for the coupled Poisson system.

    Parameters:
        Nx, Ny (int): Number of grid points in x and y directions, total grid points N = Nx * Ny.
        A1, A2 (numpy.ndarray or scipy.sparse matrix): Discretized Laplacian matrices for the electrode 
                                                       and electrolyte (size: (N, N)).
        phi1, phi2 (numpy.ndarray): Potential vectors for the electrode and electrolyte (size: (N,)).
        constrained_cells_1 (optional): Indices of reference-constrained electrode cells.
        constrained_cells_2 (optional): Indices of reference-constrained electrolyte cells.

    Returns:
        J (scipy.sparse.csr_matrix): The Jacobian matrix. If constraints are applied, the augmented matrix is returned.
    """

    if constrained_cells_1 is None:
        constrained = 0
    elif constrained_cells_2 is None:
        constrained = 1
    else:
        constrained = 2

    # **Physical Constants and Parameters**
    As = 1.64e4         # specific surface area 1/m
    E = -0.255          # reference negative electrode potential
    F = 96485           # Faraday's constant (C/mol)
    R = 8.314           # Universal gas constant (J/(mol*K))
    T = 298.15          # Temperature (K), example value
    k = 1.7e-7          # standard reaction rate constant
    c2 = 27             # V2+ concentration mol/m^3
    c3 = 1053           # V3+ concentration mol/m^3
    
    # Compute equilibrium potential and exchange current density
    j0 = (F*k) * (c3**0.5) * (c2**0.5)
    E_eq = E + (R*T/F) * np.log(c3/c2)
    
    # Define coefficients for the nonlinear terms
    a = As * j0
    b = 0.5 * F / (R * T)
    a1 = 2 * a
    a2 = -2 * a

    N = Nx * Ny

    # **Define Nonlinear Terms (Derivative Functions)**
    def df1_dphi1(eta):
        return a1 * b * np.cosh(b * eta)
    
    def df1_dphi2(eta):
        return -a1 * b * np.cosh(b * eta)
    
    def df2_dphi1(eta):
        return a2 * b * np.cosh(b * eta)
    
    def df2_dphi2(eta):
        return -a2 * b * np.cosh(b * eta)
    
    
    D11 = np.zeros((N, N))
    
    for i in range(N):
        eta = phi1[i] - phi2[i] - E_eq
        eta = min(eta, 0)  
        D11[i, i] = -df1_dphi1(eta)
    
    J11 = A1 + D11  
    
    # **J12 Matrix (Coupling Term)**
    J12 = np.zeros((N, N))
    for i in range(N):
        eta = phi1[i] - phi2[i] - E_eq
        eta = min(eta, 0)
        J12[i, i] = -df1_dphi2(eta)
    
    # **J21 Matrix (Coupling Term)**
    J21 = np.zeros((N, N))
    for i in range(N):
        eta = phi1[i] - phi2[i] - E_eq
        eta = min(eta, 0)
        J21[i, i] = -df2_dphi1(eta)
    
    D22 = np.zeros((N, N))
    for i in range(N):
        eta = phi1[i] - phi2[i] - E_eq
        eta = min(eta, 0)
        D22[i, i] = -df2_dphi2(eta)
    
    J22 = A2 + D22  
    
    # **JP Construction (Jacobian Matrix)**
    JP = np.block([
        [J11, J12],
        [J21, J22]
    ])
    

    if constrained == 0:
        return JP
    elif constrained == 1:
        n1 = len(constrained_cells_1)
        C1 = np.zeros((n1, JP.shape[0]))  
        for i in range(n1):
            C1[i, constrained_cells_1[i]] = 1  
    
        J = np.block([
            [JP, C1.T],
            [C1, np.zeros((n1, n1))]
        ])
        return J
    elif constrained == 2:
        n1 = len(constrained_cells_1)
        n2 = len(constrained_cells_2)
    
        C1 = np.zeros((n1, JP.shape[0]))  
        for i in range(n1):
            C1[i, constrained_cells_1[i]] = 1  
    
        C2 = np.zeros((n2, JP.shape[0]))  
        for i in range(n2):
            C2[i, constrained_cells_2[i]] = 1  
    
        J = np.block([
            [JP, C1.T, C2.T],  
            [C1, np.zeros((n1, n1 + n2))],
            [C2, np.zeros((n2, n1 + n2))]
        ])
        return J
