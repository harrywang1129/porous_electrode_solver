#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:50:48 2025

@author: yuhe wang
"""

#!/usr/bin/env python3
import numpy as np

def constraint_residual(phi1, phi2, constrained_cells_1, constrained_value_1, constrained_cells_2=None, constrained_value_2=None):
    """
    Computes the constraint residual for applying reference conditions to the
    electrode (`phi1`) and/or electrolyte (`phi2`).
    
    """
    if constrained_cells_2 is None or constrained_value_2 is None:
        # Only phi1 is constrained
        G = [phi1[idx] - constrained_value_1 for idx in constrained_cells_1]
    else:
        G1 = [phi1[idx] - constrained_value_1 for idx in constrained_cells_1]
        G2 = [phi2[idx - len(phi1)] - constrained_value_2 for idx in constrained_cells_2]
        G = np.concatenate([G1, G2]) 
    return G
    


   
    
    

