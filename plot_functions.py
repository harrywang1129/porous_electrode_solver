#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 01:52:57 2025

@author: yuhe wang
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sigma_kappa(Lx, Ly, Nx, Ny, sigma, kappa):
    x = [i * (Lx / (Nx - 1)) * 2e3 for i in range(Nx)]  # make the plot square
    y = [i * (Ly / (Ny - 1)) * 1e2 for i in range(Ny)]
    X, Y = np.meshgrid(x, y)
    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolor(X, Y, np.array(sigma))
    plt.title(r'$\sigma$ (S/m)', fontsize=30, fontweight='bold')
    plt.xlabel('Width (mm)', fontsize=24)
    plt.ylabel('Height (cm)', fontsize=24)
    cbar1 = plt.colorbar()
    cbar1.ax.tick_params(labelsize=18)
    plt.subplot(1,2,2)
    plt.pcolor(X, Y, np.array(kappa))
    plt.title(r'$\kappa$ (S/m)', fontsize=30, fontweight='bold')
    plt.xlabel('Width (mm)', fontsize=24)
    plt.ylabel('Height (cm)', fontsize=24)
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize=18)
    plt.show()
    
def plot_potential_results(Lx, Ly, Nx, Ny, phi_e, phi_l, eta, jr):
    x = [i * (Lx / (Nx - 1)) * 1e3 for i in range(Nx)]
    y = [i * (Ly / (Ny - 1)) * 1e2 for i in range(Ny)]
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(16,4))
    plt.subplot(1,4,1)
    plt.pcolor(X, Y, np.array(phi_e)*1e3)
    plt.title(r'$\phi_e$ (mV)', fontsize=24, fontweight='bold')
    plt.xlabel('Width (mm)', fontsize=18)
    plt.ylabel('Height (cm)', fontsize=18)
    cbar1 = plt.colorbar()
    cbar1.ax.tick_params(labelsize=18)
    plt.subplot(1,4,2)
    plt.pcolor(X, Y, np.array(phi_l)*1e3)
    plt.title(r'$\phi_l$ (mV)', fontsize=24, fontweight='bold')
    plt.xlabel('Width (mm)', fontsize=18)
    plt.ylabel('Height (cm)', fontsize=18)
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize=18)
    plt.subplot(1,4,3)
    plt.pcolor(X, Y, np.array(eta)*1e3)
    plt.title(r'$\eta$ (mV)', fontsize=24, fontweight='bold')
    plt.xlabel('Width (mm)', fontsize=18)
    plt.ylabel('Height (cm)', fontsize=18)
    cbar3 = plt.colorbar()
    cbar3.ax.tick_params(labelsize=18)
    plt.subplot(1,4,4)
    plt.pcolor(X, Y, np.array(jr))
    plt.title(r'$j$ (A/m$^2$)', fontsize=24, fontweight='bold')
    plt.xlabel('Width (mm)', fontsize=18)
    plt.ylabel('Height (cm)', fontsize=18)
    cbar4 = plt.colorbar()
    cbar4.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()    
    

