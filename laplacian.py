#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:22:35 2025

@author: yuhe wang
"""

#!/usr/bin/env python3
import numpy as np

def laplacian(Nx, Ny, dx, dy, k, dirichlet_position='none'):
    """
    Constructs a 2D Laplacian matrix with spatially varying conductivity.

    Parameters:
        Nx, Ny (int): Number of grid points in x and y directions.
        dx, dy (float): Grid spacing in x and y directions.
        k (numpy.ndarray): Conductivity field of size (Ny, Nx).
        dirichlet_position (str): 'left', 'right' or 'none', specifying Dirichlet boundary position.

    Returns:
        A: Laplacian matrix of size (Nx*Ny, Nx*Ny).

    """
    N = Nx * Ny
    A = np.zeros((N, N))   
    
    kx = np.zeros((Ny, Nx+1))  # k at vertical inter-cell faces
    ky = np.zeros((Ny+1, Nx))  # k at horizontal inter-cell faces
    
    # Compute harmonic averages for k at x-interfaces (vertical faces)
    for i in range(Ny):
        for j in range(Nx-1):
            kx[i, j+1] = 2 * k[i, j] * k[i, j+1] / (k[i, j] + k[i, j+1])

    for i in range(Ny-1):
        for j in range(Nx):
            ky[i+1, j] = 2 * k[i, j] * k[i+1, j] / (k[i, j] + k[i+1, j])

    # Apply ghost cell correction for Dirichlet boundaries
    if dirichlet_position == 'left':
        kx[:, 0] = 2 * k[:, 0] / (1.0 + 1.0)  # Approximate left boundary conductivity
    elif dirichlet_position == 'right':
        kx[:, Nx] = 2 * k[:, Nx-1] / (1.0 + 1.0)  # Approximate right boundary conductivity
    
    # Construct the Laplacian matrix
    for i in range(Ny):
        for j in range(Nx):
            idx = i * Nx + j 
 
            # Fetch interfacial conductivities
            k_right = kx[i, j+1] if j < Nx-1 else 0.0
            k_left = kx[i, j] if j > 0 else 0.0
            k_up = ky[i, j] if i > 0 else 0.0
            k_down = ky[i+1, j] if i < Ny-1 else 0.0
 
            A[idx, idx] = - (k_right + k_left) / dx**2 - (k_up + k_down) / dy**2
 
            # Adjust for Dirichlet Boundary Condition
            if j == 0 and dirichlet_position == 'left':
                A[idx, idx] -= 2 * kx[i, j] / dx**2
            elif j == Nx-1 and dirichlet_position == 'right':
                A[idx, idx] -= 2 * kx[i, j+1] / dx**2
 
            # Off-diagonal terms
            if j < Nx-1:
                A[idx, idx+1] = k_right / dx**2
            if j > 0:
                A[idx, idx-1] = k_left / dx**2
            if i > 0:
                A[idx, idx-Nx] = k_up / dy**2
            if i < Ny-1:
                A[idx, idx+Nx] = k_down / dy**2
        
    return A   
    
    

