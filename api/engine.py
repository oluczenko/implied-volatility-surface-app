import numpy as np
from scipy.special import ndtr 

def calc_d1_d2(S,K,T,r,sigma,q= 0.0):
    """
    calculate d1 and d2 for Black-Scholes formula
    """
    S= np.asarray(S, dtype=np.float64)
    K= np.asarray(K, dtype=np.float64)
    T= np.asarray(T, dtype=np.float64)
    r= np.asarray(r, dtype=np.float64)
    sigma= np.asarray(sigma, dtype=np.float64)
    q= np.asarray(q, dtype=np.float64)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def black_scholes_price(S,K,T,r,sigma,q=0.0,otype=1.0):
    """
    calculate Black-Scholes price for European options
    otype: 1.0 for call, -1.0 for put
    """
    d1, d2 = calc_d1_d2(S,K,T,r,sigma,q)
    
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    otype = np.asarray(otype, dtype=np.float64)
    return otype * (S * df_q * ndtr(otype * d1) - K * df_r * ndtr(otype * d2))


def black_scholes_vega(S,K,T,r,sigma,q=0.0):
    """
    calculate Black-Scholes vega for European options
    """
    d1, _ = calc_d1_d2(S,K,T,r,sigma,q)
    
    T = np.maximum(np.asarray(T, dtype=np.float64), 1e-6)
    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    
    return np.asarray(S, dtype=np.float64) * np.exp(-q * T) * np.sqrt(T) * pdf_d1


    