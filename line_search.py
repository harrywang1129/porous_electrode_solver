#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 01:55:44 2025

@author: yuhe wang

"""

import numpy as np
from newton_raphson import newton_raphson 
from bultervolmerclassic import bultervolmerclassic

def objective(ref_value_l, Nx, Ny, dx, dy, dV, e_left, e_right, e_top, e_bottom,
              ref_value_e, l_left, l_right, l_top, l_bottom, c3, c2, I,
              aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
              reference_cells_e, reference_cells_l, lambda_vec, case_index):
    """Solves the Newton-Raphson system and computes charge balance error."""
    max_iter, iter_count, tol, omega = 1, 0, 1e-6, 1.0
    N = Nx * Ny
    while iter_count < max_iter:
        iter_count += 1
        x, _, _ = newton_raphson(Nx, Ny, dx, dy, sigma, kappa, phi_ev_old, phi_lv_old,
                         e_top, e_bottom, e_left, e_right, l_top, l_bottom, 
                         l_left, ref_value_l if case_index == 2 else l_right, case_index,
                         reference_cells_e=reference_cells_e if case_index == 1 else None,
                         reference_cells_l=reference_cells_l if case_index == 1 else None,
                         ref_value_e=ref_value_e if case_index == 1 else None,
                         ref_value_l=ref_value_l if case_index == 1 else None,
                         lambda_vec=lambda_vec if case_index == 1 else None)

        phi_ev_old = x[:N]
        phi_lv_old = x[N:2*N]
        phi_e = phi_ev_old.reshape((Ny, Nx))
        phi_l = phi_lv_old.reshape((Ny, Nx))      
        # phi_e = phi_ev_old.reshape((Ny, Nx), order='F').T
        # phi_l = phi_lv_old.reshape((Ny, Nx), order='F').T       
        aj, _, _ = bultervolmerclassic(phi_e, phi_l, c3, c2)
        aj = omega * aj + (1.0 - omega) * aj_old
        res_norm = np.linalg.norm(aj - aj_old) / (np.linalg.norm(aj_old) + 1e-12)
        aj_old = aj
        
        if res_norm < tol:
            break
        
    charge_total = np.sum(np.abs(aj) * dV)
    error = np.abs(charge_total - I) / I
    return error, iter_count


def compute_gradient(ref_value_l, delta, Nx, Ny, dx, dy, dV,
                     e_left, e_right, e_top, e_bottom, ref_value_e, l_left, l_right, l_top,
                     l_bottom, c3, c2, I, aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
                     reference_cells_e, reference_cells_l, lambda_vec, case_index):
    """Computes the numerical gradient of the objective function."""
    error_current, iter1 = objective(ref_value_l, Nx, Ny, dx, dy, dV,
                                     e_left, e_right, e_top, e_bottom, ref_value_e,
                                     l_left, l_right, l_top, l_bottom, c3, c2, I,
                                     aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
                                     reference_cells_e, reference_cells_l, lambda_vec, case_index)
    error_delta, iter2 = objective(ref_value_l + delta, Nx, Ny, dx, dy, dV,
                                   e_left, e_right, e_top, e_bottom, ref_value_e,
                                   l_left, l_right, l_top, l_bottom, c3, c2, I,
                                   aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
                                   reference_cells_e, reference_cells_l, lambda_vec, case_index)
    grad = (error_delta - error_current) / delta
    return grad, (iter1 + iter2)

        
def update(ref_value_l_old, Nx, Ny, dx, dy, dV,
           e_left, e_right, e_top, e_bottom, ref_value_e, l_left, l_right,
           l_top, l_bottom, c3, c2, I, aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
           reference_cells_e, reference_cells_l, lambda_vec, case_index):
    """Performs a line search update on ref_value_l using gradient descent."""

    # Line search parameters
    alpha, rho, c = 0.001, 0.2, 0.1
    max_iter, delta = 100, 1.0e-6
    
    grad, iter1 = compute_gradient(ref_value_l_old, delta, Nx, Ny, dx, dy, dV,
                                   e_left, e_right, e_top, e_bottom, ref_value_e,
                                   l_left, l_right, l_top, l_bottom, c3, c2, I,
                                   aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
                                   reference_cells_e, reference_cells_l, lambda_vec, case_index)
    
    ref_value_l_new = ref_value_l_old
    iter_p, it_count = 0, 0
    
    for it in range(max_iter):
        it_count+= 1
        ref_value_l_new = ref_value_l_old - alpha * grad
        error_new, iter2 = objective(ref_value_l_new, Nx, Ny, dx, dy, dV,
                                     e_left, e_right, e_top, e_bottom, ref_value_e,
                                     l_left, l_right, l_top, l_bottom, c3, c2, I,
                                     aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
                                     reference_cells_e, reference_cells_l, lambda_vec, case_index)
        error_current, iter3 = objective(ref_value_l_old, Nx, Ny, dx, dy, dV,
                                         e_left, e_right, e_top, e_bottom, ref_value_e,
                                         l_left, l_right, l_top, l_bottom, c3, c2, I,
                                         aj_old, phi_ev_old, phi_lv_old, sigma, kappa,
                                         reference_cells_e, reference_cells_l, lambda_vec, case_index)
        
        iter_p += (iter2 + iter3)
        
        if error_new <= error_current + c * alpha * grad * (-grad):
            break
        else:
            alpha = rho * alpha
            
    iter_p += iter1
    if it == max_iter - 1:
        print("Line search did not converge within max iterations.")
        
    return ref_value_l_new, it_count, iter_p