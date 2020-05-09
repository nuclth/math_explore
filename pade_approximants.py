import math
import numpy as np

#  get array of taylor expansion coefficients
def get_array_taylor_coeff(order, taylor_coeff_func):
    taylor_coeffs = np.zeros(order)
    
    for n in range (0, order):
        taylor_coeffs[n] = taylor_coeff_func(n)
    
    return taylor_coeffs

# function to return the sum of a power series in x^n
def get_polynomial(coeffs, x, order):    
    pade_sum = 0
    
    for idx in range(0, order+1):
        term = coeffs[idx] * math.pow(x, idx)
        pade_sum += term

    return pade_sum

# create the coupled equation matrix to determine diagonal [n,n] pade approximant coefficients 
def create_pade_matrix(taylor_coeffs, order):
    
    pade_matrix = np.zeros((order+1, order+1))
    
    qIdxStart = int(order/2 + 1)
        
    # set diagonal element
    for idx in range(0, qIdxStart):
        pade_matrix[idx,idx] = -1
    
    for row in range (0, order+1):
        taylorIdx = row - 1
                        
        for qIdx in range (0, row):
            qMatrixIdx = qIdxStart + qIdx
            
            if (qMatrixIdx > order):
                continue
            
            pade_matrix[row, qMatrixIdx] = taylor_coeffs[taylorIdx]
            taylorIdx -= 1
            
                        
    return pade_matrix   

# return the diagonal [n,n] pade approximant
def pade_approximant(z, order, taylor_coeff_func):

    if order % 2 != 0:
        print("Error: Pade approximant function can currently handle only even orders")
        SystemExit    
    
    pade_order = int(order/2)

    taylor_coeffs = get_array_taylor_coeff(order+1, taylor_coeff_func)
    pade_matrix = create_pade_matrix(taylor_coeffs, order)
    a = pade_matrix
    b = -1 * taylor_coeffs
    pade_coeffs = np.linalg.solve(a, b)

    p_coeffs = pade_coeffs[0:pade_order+1]
    q_coeffs = np.zeros(pade_order + 1)
    q_coeffs[0] = 1
    qIdx = 1
    for idx in range(pade_order + 1, order+1):
        q_coeffs[qIdx] = pade_coeffs[idx]
        qIdx += 1

    pade_numer = get_polynomial(p_coeffs, z, pade_order)
    pade_denom = get_polynomial(q_coeffs, z, pade_order)
    
    return (pade_numer / pade_denom)