#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:27:25 2025

@author: yuhe wang
"""

def laplacian_multiply(phi, A, Nx, Ny, dx, dy, g_top, g_bottom, g_left, g_right, k=None, dirichlet_position='none'):
    """
    Computes the residual R = A * phi + boundary contributions, handling Neumann and Dirichlet boundary conditions.

    Parameters:
        - phi (numpy.ndarray): Solution vector of length Nx*Ny.
        - A (numpy.ndarray or scipy.sparse matrix): Laplacian matrix of size (Nx*Ny, Nx*Ny).
        - g_top, g_bottom, g_left, g_right (float): Neumann boundary conditions.
        - k: Conductivity field (optional, defaults to ones(Ny, Nx))
        - dirichlet_position: Dirichlet boundary ('none', 'left', 'right')

    Returns:
        - R (numpy.ndarray): Residual vector of size (Nx*Ny).

    """
    R = A @ phi

    # Add Dirichlet condition contribution
    if dirichlet_position == 'left':
        for i in range(Ny):
            idx = i * Nx  
            k_left = k[i][0]
            R[idx] += (2 * k_left * g_left) / (dx*dx)
    elif dirichlet_position == 'right':
        for i in range(Ny):
            idx = i * Nx + (Nx - 1)
            k_right = k[i][Nx-1]
            R[idx] += (2 * k_right * g_right) / (dx*dx)
    
    for i in range(Ny):
        for j in range(Nx):
            idx = i * Nx + j
            if (dirichlet_position == 'left' and j == 0) or (dirichlet_position == 'right' and j == Nx-1):
                continue
            
            if i == 0 and j == 0:
                R[idx] -= g_top / dy + g_left / dx
            elif i == Ny-1 and j == 0:
                R[idx] += g_bottom / dy - g_left / dx
            elif i == 0 and j == Nx-1:
                R[idx] -= g_top / dy - g_right / dx
            elif i == Ny-1 and j == Nx-1:
                R[idx] += g_bottom / dy + g_right / dx
                
            if i == 0 and 0 < j < Nx-1:
                R[idx] -= g_top / dy
            elif i == Ny-1 and 0 < j < Nx-1:
                R[idx] += g_bottom / dy
            if j == 0 and 0 < i < Ny-1:
                R[idx] -= g_left / dx
            elif j == Nx-1 and 0 < i < Ny-1:
                R[idx] += g_right / dx
    return R

