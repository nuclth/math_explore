import math
import scipy.special

def optimal_Newton_E0(M, e):
    
    if e >=1:
        print("Cannot use initial E0 function with e >= 1")
        SystemExit
    
    if M >= 0 and M < 0.25:
        E_0 = M + e*e * (math.pow(6.*M, 1./3.) - M)
    elif M >= 0.25 and M <= math.pi:
        numer = e * math.sin(M)
        denom = 1. - math.sin(M + e) + math.sin(M)
        E_0 = M + (numer/denom)
    else:
        print("M out of bounds")
        SystemExit
    
    return E_0

def Newton_Raphson(e, M, desired_accuracy):
    
    E_0 = optimal_Newton_E0(M, e)
    
    E_i = E_0
    E_j = Newton_Raphson_Iteration(E_i, e, M)
    
    while (abs(E_i - E_j) > desired_accuracy):
        E_i = E_j
        E_j = Newton_Raphson_Iteration(E_i, e, M)
    
    return E_j
    

def Newton_Raphson_Iteration(E_i, e, M):
    func_deriv_quotient_numer = E_i - e * math.sin(E_i) - M
    func_deriv_quotient_denom = 1. - e * math.cos(E_i)
    func_deriv_quotient = func_deriv_quotient_numer / func_deriv_quotient_denom
    
    E_j = E_i - func_deriv_quotient
    
    return E_j


# get nth taylor expansion coefficient
def taylor_coeff_func_eccentricity(n, M):
    if n == 0:
        return M
    else:
        prefac = 1. / (math.pow(2., n -1) * math.factorial(n))

        running_sum = 0
        
        for k in range(0, math.floor(n/2) + 1):
            term1 = math.pow(-1., k) * scipy.special.binom(n, k)
            term2 = math.pow(n - 2.*k, n-1.) * math.sin((n - 2.*k) * M)
            running_sum += (term1 * term2)
            
        return running_sum * prefac