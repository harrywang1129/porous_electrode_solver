#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:29:49 2025

@author: yuhe wang
"""
import numpy as np
from scipy.sparse import eye, csr_matrix, bmat
from scipy.sparse.linalg import minres, bicgstab, spsolve, splu, LinearOperator, lsmr

from laplacian import laplacian
from laplacian_multiply import laplacian_multiply
from jacobian import jacobian
from source_residual import source_residual
from constraint_residual import constraint_residual

def newton_raphson(Nx, Ny, dx, dy, sigma, kappa, phi_ev_old, phi_lv_old,
                   e_top, e_bottom, e_left, e_right, l_top, l_bottom, l_left, l_right,
                   case_index, **kwargs):
    """
    Newton–Raphson solver for the coupled potential equations.
    
    Required Inputs:
      Nx, Ny      - Number of grid points in x and y directions.
      dx, dy      - Grid spacing.
      sigma, kappa - Conductivity fields for electrode and electrolyte (2D arrays of shape (Ny, Nx)).
      phi_ev_old, phi_lv_old - Initial guesses (flattened arrays/lists of length Nx*Ny).
      e_top, e_bottom, e_left, e_right - Neumann BCs for the electrode.
      l_top, l_bottom, l_left, l_right - Neumann BCs for the electrolyte.
      case_index  - Strategy selector:
                    1: LCM & Picard (requires: reference_cells_e, reference_cells_l, ref_value_e, ref_value_l, lambda)
                    2: DSM & Picard (uses Dirichlet BCs: electrode left, electrolyte right)
                    3: LCM & Fully-Coupled (only electrode reference enforced; requires reference_cells_e, ref_value_e, lambda)
                    4: DSM & Fully-Coupled (no reference parameters)
                    5: Fully-Coupled (no reference constraints)
                    6: Potentiostatic with potential sweeping (electrode left Dirichlet, electrolyte right Dirichlet)
                    7: Potentiostatic with current sweeping (electrode left Dirichlet, electrolyte Neumann)
                    
    Optional parameters (via kwargs):
      'reference_cells_e' - list of reference cell indices for electrode (0-based)
      'reference_cells_l' - list of reference cell indices for electrolyte (0-based)
      'ref_value_e'       - electrode reference potential (scalar)
      'ref_value_l'       - electrolyte reference potential (scalar)
      'lambda'            - initial Lagrange multipliers (list or array)
      
    Returns:
      x: solution vector containing electrode and electrolyte potentials (and lambda if applicable)
      iter_nr: number of Newton–Raphson iterations
      residual_nr: final relative residual
    """
    
    # Default optional parameters
    reference_cells_e = kwargs.get('reference_cells_e', np.array([]))
    reference_cells_l = kwargs.get('reference_cells_l', np.array([]))
    ref_value_e = kwargs.get('ref_value_e', None)
    ref_value_l = kwargs.get('ref_value_l', None)
    lambda_vec = kwargs.get('lambda_vec', np.array([]))
    
   # Validate required parameters based on `case_index`
    if case_index == 1:
        if len(reference_cells_e) == 0 or len(reference_cells_l) == 0 or \
           ref_value_e is None or ref_value_l is None or len(lambda_vec) == 0:
            raise ValueError("For case 1, all reference parameters are required.")
    
    if case_index == 3:
        if len(reference_cells_e) == 0 or ref_value_e is None or len(lambda_vec) == 0:
            raise ValueError("For case 3, reference_cells_e, ref_value_e, and lambda are required.")


    max_iter_nr = 200
    max_iter_solver = 100
    tol_nr = 1e-6
    tol_solver = 1e-6
    tol_residualchange = 1e-6
    iter_nr = 0
    R_old = 0.0
    R1_old = 0.0
    R2_old = 0.0
    N = Nx * Ny

    # Construct Laplacian matrices based on the case
    if case_index in [1,3,5]:
        Ae = laplacian(Nx, Ny, dx, dy, sigma)
        Al = laplacian(Nx, Ny, dx, dy, kappa)
    elif case_index in [2,6]:
        Ae = laplacian(Nx, Ny, dx, dy, sigma, 'left')
        Al = laplacian(Nx, Ny, dx, dy, kappa, 'right')
    # elif case_index in [3,5]:
    #     Ae = laplacian(Nx, Ny, dx, dy, sigma)
    #     Al = laplacian(Nx, Ny, dx, dy, kappa)
    elif case_index in [4,7]:
        Ae = laplacian(Nx, Ny, dx, dy, sigma, 'left')
        Al = laplacian(Nx, Ny, dx, dy, kappa)    
        
    if case_index in [1]:    
        while iter_nr < max_iter_nr:
            J = jacobian(Nx, Ny, Ae, Al, phi_ev_old, phi_lv_old, reference_cells_e, reference_cells_l)
            J[ :N, N:2*N] = 0.0
            J[N:2*N,  :N] = 0.0
            n1 = len(reference_cells_e)
            n2 = len(reference_cells_l)
            J11 = J[ :N, :N]
            J22 = J[N:2*N, N:2*N]
            C1 = J[2*N:2*N+n1, :N]
            C2 = J[2*N+n1:, N:2*N]
            J1 = np.block([[J11, C1.T], [C1, np.zeros((n1, n1))]])  # J1 = [J11, C1'; C1, zeros(n1)];
            J2 = np.block([[J22, C2.T], [C2, np.zeros((n2, n2))]])  # J2 = [J22, C2'; C2, zeros(n2)]; 
            
            Re = laplacian_multiply(phi_ev_old, Ae, Nx, Ny, dx, dy, e_top, e_bottom, e_left, e_right)
            Rl = laplacian_multiply(phi_lv_old, Al, Nx, Ny, dx, dy, l_top, l_bottom, l_left, l_right)
            RRe, RRl = source_residual(Nx, Ny, phi_ev_old, phi_lv_old, reference_cells_e, lambda_vec, reference_cells_l)
            G = constraint_residual(phi_ev_old, phi_lv_old, reference_cells_e, ref_value_e, reference_cells_l, ref_value_l)
            R1 = np.concatenate([Re + RRe, G[:n1]])  # R1 = [Re + RRe ; G(1:n1)]
            R2 = np.concatenate([Rl + RRl, G[n1:]])  # R2 = [Rl + RRl ; G(n1+1:end)]
            
            # Rescale
            A1 = J1[:N, :N]  
            B1 = J1[N:, :N] 
            alpha1 = np.sqrt(np.max(np.abs(np.diag(A1))))  
            B1_scaled = alpha1 * B1  
            J1S = np.block([[A1, B1_scaled.T], [B1_scaled, np.zeros((B1.shape[0], B1.shape[0]))]])  
            R1S = np.concatenate([R1[:A1.shape[0]], R1[A1.shape[0]:] * alpha1])  
            J1S = csr_matrix(J1S)
            
            # delta1_scaled, _ = bicgstab(J1S, -R1S, atol=tol_solver, maxiter=max_iter_solver)
            delta1_scaled = spsolve(J1S, -R1S)
            delta1 = np.concatenate([delta1_scaled[:A1.shape[0]], delta1_scaled[A1.shape[0]:] * alpha1]).reshape(-1,1)
            
            lambda_n1 = lambda_vec[:n1]
            x1_old = np.concatenate([phi_ev_old, lambda_n1]) 
            x1 = x1_old + delta1 
            
            A2 = J2[:N, :N]  # J2(1:N, 1:N)
            B2 = J2[N:, :N]  # J2(N+1:end, 1:N)
            alpha2 = np.sqrt(np.max(np.abs(np.diag(A2))))
            B2_scaled = alpha2 * B2
            J2S = np.block([[A2, B2_scaled.T], [B2_scaled, np.zeros((B2.shape[0], B2.shape[0]))]])
            R2S = np.concatenate([R2[:A2.shape[0]], R2[A2.shape[0]:] * alpha2])
            J2S = csr_matrix(J2S)

            # delta2_scaled, _ = bicgstab(J2S, -R2S, atol=tol_solver, maxiter=max_iter_solver)
            delta2_scaled = spsolve(J2S, -R2S)
            delta2 = np.concatenate([delta2_scaled[:A2.shape[0]], delta2_scaled[A2.shape[0]:] * alpha2]).reshape(-1,1)
          
            x2_old = np.concatenate([phi_lv_old, lambda_vec[n1:]])
            x2 = x2_old + delta2
            x = np.concatenate([x1[:N], x2[:N], x1[N:], x2[N:]])  
            
            phi_ev_old = x[:N]  
            phi_lv_old = x[N:2*N]  
            lambda_vec = x[2*N:]
            
            residual_nr_1 = np.linalg.norm(delta1) / np.linalg.norm(x1)
            residual_nr_2 = np.linalg.norm(delta2) / np.linalg.norm(x2)
            if np.linalg.norm(R1_old) != 0:
                residual_change_1 = np.linalg.norm(R1 - R1_old) / np.linalg.norm(R1_old)
            else:
                    residual_change_1 = np.linalg.norm(R1)

            if np.linalg.norm(R2_old) != 0:
                residual_change_2 = np.linalg.norm(R2 - R2_old) / np.linalg.norm(R2_old)
            else:
                    residual_change_2 = np.linalg.norm(R2)

            R1_old = R1
            R2_old = R2
            
            iter_nr += 1            
            residual_nr = (residual_nr_1 + residual_nr_2) * 0.5           
            # print(f"Newton-Raphson in {iter_nr} iterations with relative error {residual_nr:.6e}")
            if residual_nr_1 < tol_nr and residual_nr_2 < tol_nr:
                break
            elif residual_change_1 < tol_residualchange and residual_change_2 < tol_residualchange:
                break 
            
    elif case_index in [2]:   
         while iter_nr < max_iter_nr:
             J = jacobian(Nx, Ny, Ae, Al, phi_ev_old, phi_lv_old)
             J[:N, N:2*N] = 0
             J[N:2*N, :N] = 0
             
             Re = laplacian_multiply(phi_ev_old, Ae, Nx, Ny, dx, dy, e_top, e_bottom, e_left, e_right, sigma, 'left')
             Rl = laplacian_multiply(phi_lv_old, Al, Nx, Ny, dx, dy, l_top, l_bottom, l_left, l_right, kappa, 'right')
             RRe, RRl = source_residual(Nx, Ny, phi_ev_old, phi_lv_old)
             R = np.concatenate([Re + RRe, Rl + RRl])

             J = csr_matrix(J)

             delta, _ = bicgstab(J, -R, atol=tol_solver, maxiter=max_iter_solver)             
             delta = delta.reshape(-1, 1)  

             x_old = np.concatenate([phi_ev_old, phi_lv_old])
             x = x_old + delta          

             phi_ev_old = x[:N]
             phi_lv_old = x[N:2*N]
             
             residual_nr = np.linalg.norm(delta) / np.linalg.norm(x)
             
             if np.linalg.norm(R_old) != 0:
                 residual_change = np.linalg.norm(R - R_old) / np.linalg.norm(R_old)
             else:
                 residual_change = np.linalg.norm(R)
             
             R_old = R
             iter_nr += 1
             
             if residual_nr < tol_nr or residual_change < tol_residualchange:
                 break
             
    elif case_index in [3]:
        while iter_nr < max_iter_nr:
            J = jacobian(Nx, Ny, Ae, Al, phi_ev_old, phi_lv_old, reference_cells_e)
            Re = laplacian_multiply(phi_ev_old, Ae, Nx, Ny, dx, dy, e_top, e_bottom, e_left, e_right)
            Rl = laplacian_multiply(phi_lv_old, Al, Nx, Ny, dx, dy, l_top, l_bottom, l_left, l_right)            
            RRe, RRl = source_residual(Nx, Ny, phi_ev_old, phi_lv_old, reference_cells_e, lambda_vec)
            G = constraint_residual(phi_ev_old, phi_lv_old, reference_cells_e, ref_value_e)            
            R = np.concatenate([Re + RRe, Rl + RRl, G])
            
            # rescale
            A = J[:2*N, :2*N]
            B = J[2*N:, :2*N]          
            alpha = np.sqrt(np.max(np.abs(np.diag(A))))
            B_scaled = alpha * B
            JS = np.block([[A, B_scaled.T], [B_scaled, np.zeros((B.shape[0], B.shape[0]))]])
            RS = np.concatenate([R[:A.shape[0]], R[A.shape[0]:] * alpha])
            
            # linear solve            
            JS = csr_matrix(JS) 
            
            # preconditioner
            lu = splu(JS)            
            def preconditioner(x):
                return lu.solve(x)          
            M = LinearOperator(JS.shape, matvec=preconditioner)

            delta_scaled, _ = bicgstab(JS, -RS, atol=tol_solver, maxiter=max_iter_solver, M=M)
            # delta_scaled, exit_code = gmres(JS, -RS, restart=50, atol=tol_solver, maxiter=max_iter_solver, M=M)
            # delta_scaled = spsolve(JS, -RS)
            
            delta = np.concatenate([delta_scaled[:A.shape[0]], delta_scaled[A.shape[0]:] * alpha]).reshape(-1,1)
            
            x_old = np.concatenate([phi_ev_old, phi_lv_old, lambda_vec])
            x = x_old + delta
            
            phi_ev_old = x[:N]
            phi_lv_old = x[N:2*N]
            lambda_vec = x[2*N:]
            
            residual_nr = np.linalg.norm(delta) / np.linalg.norm(x)
            
            if np.linalg.norm(R_old) != 0:
                residual_change = np.linalg.norm(R - R_old) / np.linalg.norm(R_old)
            else:
                residual_change = np.linalg.norm(R)
            
            R_old = R
            iter_nr += 1
            print(f"Newton-Raphson in {iter_nr} iterations with relative error {residual_nr:.6e}")
            
            if residual_nr < tol_nr:
                print(f"Newton-Raphson converged in {iter_nr} iterations with relative error {residual_nr:.6e}")
                break
            elif residual_change < tol_residualchange:
                print(f"Residual change stopped reducing at iteration {iter_nr}. Change: {residual_change:.6e}, Residual norm: {np.linalg.norm(R):.6e}")
                break 
            
    elif case_index in [4,7]:
        while iter_nr < max_iter_nr:
            J = jacobian(Nx, Ny, Ae, Al, phi_ev_old, phi_lv_old)            
            Re = laplacian_multiply(phi_ev_old, Ae, Nx, Ny, dx, dy, e_top, e_bottom, e_left, e_right, sigma, 'left')
            Rl = laplacian_multiply(phi_lv_old, Al, Nx, Ny, dx, dy, l_top, l_bottom, l_left, l_right)           
            RRe, RRl = source_residual(Nx, Ny, phi_ev_old, phi_lv_old)
            R = np.concatenate([Re + RRe, Rl + RRl])
            
            # linear solve            
            J = csr_matrix(J) 
            
            # preconditioner
            lu = splu(J)            
            def preconditioner(x):
                return lu.solve(x)          
            M = LinearOperator(J.shape, matvec=preconditioner)
            delta, _ = bicgstab(J, -R, atol=tol_solver, maxiter=max_iter_solver, M=M)
            
            x_old = np.concatenate([phi_ev_old, phi_lv_old])
            delta = delta.reshape(-1, 1)  
            x = x_old + delta
            
            phi_ev_old = x[:N]
            phi_lv_old = x[N:2*N]
            
            residual_nr = np.linalg.norm(delta) / np.linalg.norm(x)
            
            if np.linalg.norm(R_old) != 0:
                residual_change = np.linalg.norm(R - R_old) / np.linalg.norm(R_old)
            else:
                residual_change = np.linalg.norm(R)
            
            R_old = R
            iter_nr += 1
            
            print(f"Newton-Raphson in {iter_nr} iterations with relative error {residual_nr:.6e}")
            
            if residual_nr < tol_nr:
                print(f"Newton-Raphson converged in {iter_nr} iterations with relative error {residual_nr:.6e}")
                break
            elif residual_change < tol_residualchange:
                print(f"Residual change stopped reducing at iteration {iter_nr}. Change: {residual_change:.6e}, Residual norm: {np.linalg.norm(R):.6e}")
                break  

    elif case_index in [5]:
        while iter_nr < max_iter_nr:    
            J = jacobian(Nx, Ny, Ae, Al, phi_ev_old, phi_lv_old)           
            Re = laplacian_multiply(phi_ev_old, Ae, Nx, Ny, dx, dy, e_top, e_bottom, e_left, e_right)
            Rl = laplacian_multiply(phi_lv_old, Al, Nx, Ny, dx, dy, l_top, l_bottom, l_left, l_right)            
            RRe, RRl = source_residual(Nx, Ny, phi_ev_old, phi_lv_old)
            R = np.concatenate([Re + RRe, Rl + RRl])
            
            # linear solve            
            J = csr_matrix(J) 
            
            tol_minres = max(1e-6, min(1e-3, np.linalg.norm(R, np.inf) / 1000))
            
            # preconditioning MINRES                           
            # lambda_ = 1e-6
            # I = eye(J.shape[1])
            
            # A_reduced = J - lambda_**2 * I
            # b_reduced = -R
            
            # alpha = 0.3 + 0.7 * (np.count_nonzero(J.diagonal()) / J.nnz)
            # beta = 1.0 - alpha
            # mask = (np.abs(J).sum(axis=1) > 1e-10).astype(float)  # Ensure boolean to float conversion 
            # mask = np.ravel(mask)   # to 1D 
            # diag_J = np.ravel(np.abs(J.diagonal()))
            # row_sum_J = np.ravel(np.abs(J).sum(axis=1))   # row-wise sum
            # M_diag = mask * (alpha * diag_J + beta * row_sum_J) + lambda_
            # M = diags(M_diag.ravel(), 0, shape=(J.shape[0], J.shape[0]), format="csr")        
            # delta, exitcode = minres(A_reduced, b_reduced, x0=None, rtol=tol_minres, maxiter=max_iter_solver, M=M)    
            # delta, exitcode = minres(J, -R, x0=None, rtol=tol_minres, maxiter=max_iter_solver)    
            # delta, exitcode = minres(A_reduced, b_reduced, x0=None, rtol=tol_minres, maxiter=max_iter_solver)    
            
            # Least-Sqaures with Tikhonov Regularization                      
            lambda_ = 0.1
            I = eye(J.shape[1])
            
            J_reg = bmat([[J], [np.sqrt(lambda_) * I]], format='csr')
            R_reg = np.concatenate([-R.ravel(), np.zeros(I.shape[0])])
            delta = lsmr(J_reg, R_reg)[0]
                                              
            x_old = np.concatenate([phi_ev_old, phi_lv_old])
            delta = delta.reshape(-1, 1)  
            x = x_old + delta
            
            phi_ev_old = x[:N]
            phi_lv_old = x[N:2*N]
            
            residual_nr = np.linalg.norm(delta) / np.linalg.norm(x)
            
            if np.linalg.norm(R_old) != 0:
                residual_change = np.linalg.norm(R - R_old) / np.linalg.norm(R_old)
            else:
                residual_change = np.linalg.norm(R)
            
            R_old = R
            iter_nr += 1
            
            print(f"Newton-Raphson at {iter_nr} iterations with relative error {residual_nr:.6e}")
            
            if residual_nr < tol_nr:
                print(f"Newton-Raphson converged in {iter_nr} iterations with relative error {residual_nr:.6e}")
                break
            elif residual_change < tol_residualchange:
                print(f"Residual change stopped reducing at iteration {iter_nr}. Change: {residual_change:.6e}, Residual norm: {np.linalg.norm(R):.6e}")
                break 
    elif case_index in [6]:
        while iter_nr < max_iter_nr:             
            J = jacobian(Nx, Ny, Ae, Al, phi_ev_old, phi_lv_old)           
            Re = laplacian_multiply(phi_ev_old, Ae, Nx, Ny, dx, dy, e_top, e_bottom, e_left, e_right, sigma, 'left')
            Rl = laplacian_multiply(phi_lv_old, Al, Nx, Ny, dx, dy, l_top, l_bottom, l_left, l_right, kappa, 'right')            
            RRe, RRl = source_residual(Nx, Ny, phi_ev_old, phi_lv_old)
            R = np.concatenate([Re + RRe, Rl + RRl])
            
            delta = np.linalg.solve(J, -R)
            
            x_old = np.concatenate([phi_ev_old, phi_lv_old])
            delta = delta.reshape(-1, 1)  
            x = x_old + delta
            
            phi_ev_old = x[:N]
            phi_lv_old = x[N:2*N]
            
            residual_nr = np.linalg.norm(delta) / np.linalg.norm(x)
            
            if np.linalg.norm(R_old) != 0:
                residual_change = np.linalg.norm(R - R_old) / np.linalg.norm(R_old)
            else:
                residual_change = np.linalg.norm(R)
            
            R_old = R
            iter_nr += 1
            
            print(f"Newton-Raphson converged in {iter_nr} iterations with relative error {residual_nr:.6e}")
            
            if residual_nr < tol_nr:
                print(f"Newton-Raphson converged in {iter_nr} iterations with relative error {residual_nr:.6e}")
                break
            elif residual_change < tol_residualchange:
                print(f"Residual change stopped reducing at iteration {iter_nr}. Change: {residual_change:.6e}, Residual norm: {np.linalg.norm(R):.6e}")
                break   
             
                          
    return x, iter_nr, residual_nr

             
         
         
         
         
         
        