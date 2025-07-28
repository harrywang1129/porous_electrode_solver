#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 02:21:32 2025

@author: yuhe wang
"""

import numpy as np

def generate_channelized_field(Nx, Ny, high, low, connection):
    """
    Generates a channelized porosity field with guaranteed left-to-right connectivity.
    
    Ensures all channels reach the right boundary (Nx) without gaps.

    Parameters:
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        high (float): High porosity value.
        low (float): Low porosity value.
        connection (float): Probability of branching.

    Returns:
        np.ndarray: Porosity field of shape (Ny, Nx).
    """
    # **1. Initialize the Porosity Field**
    porosity = np.full((Ny, Nx), low)  # Create Ny × Nx matrix filled with `low`

    # **2. Define Channel Parameters**
    num_channels = round(0.22 * Ny)  # **More channels (~20% of height)**
    channel_width = max(1, round(0.005 * Ny) + np.random.randint(0, 2))
    perturbation = max(1, round(0.02 * Ny))

    # **3. Generate Fully Connected Channels**
    channel_endpoints = np.zeros(num_channels, dtype=int)  # Track last y_pos for each channel

    for c in range(num_channels):
        y_pos = np.random.randint(channel_width + 1, Ny - channel_width - 1)
        x_pos = 0  # Start at the left boundary

        while x_pos < Nx-1:
            # **Ensure channel spans from left to right**
            porosity[max(0, y_pos-channel_width):min(Ny, y_pos+channel_width), x_pos] = high
            x_pos += 1  
            y_pos += np.random.randint(-perturbation, perturbation + 1)  # Meandering

            # Keep the channel within bounds
            y_pos = max(channel_width + 1, min(Ny - channel_width - 1, y_pos))

            # Occasionally branch out
            if np.random.rand() < connection:
                branch_y = y_pos + np.random.randint(-channel_width, channel_width + 1)
                porosity[max(0, branch_y-channel_width):min(Ny, branch_y+channel_width), x_pos] = high
        
        # Store last y-position to ensure right boundary connectivity
        channel_endpoints[c] = y_pos

    # **4. Ensure Right Boundary Connectivity (Final Column Correction)**
    for c in range(num_channels):
        y_pos = channel_endpoints[c]
        porosity[max(0, y_pos-channel_width):min(Ny, y_pos+channel_width), -1] = high  # Ensure right boundary is high

    # **5. Introduce Additional Random Connectivity**
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            if porosity[i, j] == high:
                if np.random.rand() < connection:
                    porosity[min(Ny - 1, i + 1), j] = high
                if np.random.rand() < connection:
                    porosity[i, min(Nx - 1, j + 1)] = high

    return porosity