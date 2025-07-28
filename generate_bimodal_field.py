#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:10:37 2025

@author: yuhe wang
"""

#!/usr/bin/env python3
import random
import numpy as np

def generate_bimodal_field(Nx, Ny, Lx, Ly, high, low, connection):
    """
    Generates a bimodal porosity field with a lattice pattern.

    Parameters:
        Nx, Ny (int): Number of grid points in x and y directions.
        Lx, Ly (float): Physical dimensions.
        high, low (float): High and low porosity values.
        connection (float): Probability of high-porosity regions being connected.

    Returns:
        field (numpy.ndarray): A (Ny, Nx) array representing the porosity field.

    """

    porosity = np.full((Ny, Nx), low, dtype=float)

    # Define hexagonal pattern size relative to grid size
    hex_size_x = max(Nx // 10, 1)  # Ensure at least 1 cell wide
    hex_size_y = max(Ny // 10, 1)  # Ensure at least 1 cell high

    # Create hexagonal pattern
    for i in range(Ny):
        for j in range(Nx):
            if (j // hex_size_x) % 2 == (i // hex_size_y) % 2:
                porosity[i, j] = high

    # Add connectivity
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            if porosity[i, j] == high:
                if random.random() < connection:
                    porosity[i + 1, j] = high
                if random.random() < connection:
                    porosity[i, j + 1] = high

    return porosity
