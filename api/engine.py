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

def _bjerksund_stensland_call_scalar(S, K, T, r, q, sigma):
    """
    Calculate the Bjerksund-Stensland American call option price for a single set of parameters.
    """
    b = r - q

    if b >= r:
        return black_scholes_price(S, K, T, r, sigma, q, otype=1.0)

    sigma2 = sigma ** 2
    sqrt_T = np.sqrt(T)

    beta = (0.5 - b / sigma2) + np.sqrt((b / sigma2 - 0.5) ** 2 + 2.0 * r / sigma2)
    B_infinity = (beta / (beta - 1.0)) * K

    B_0 = max(K, (r / max(q, 1e-12)) * K)

    h = -(b * T + 2.0 * sigma * sqrt_T) * (B_0 / max(B_infinity - B_0, 1e-12))
    I = B_0 + (B_infinity - B_0) * (1.0 - np.exp(h))

    if S >= I:
        return S - K

    alpha = (I - K) * (I ** (-beta))

    def _phi(S_val, T_val, gamma, H_val, I_val):
        lambda_val = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma2
        kappa = 2.0 * b / sigma2 + (2.0 * gamma - 1.0)

        d1 = -(np.log(S_val / H_val) + (b + (gamma - 0.5) * sigma2) * T_val) / (sigma * sqrt_T)
        d2 = -(np.log((I_val ** 2) / (S_val * H_val)) + (b + (gamma - 0.5) * sigma2) * T_val) / (sigma * sqrt_T)

        return np.exp(lambda_val * T_val) * (S_val ** gamma) * (ndtr(d1) - ((I_val /S_val) ** kappa) * ndtr(d2))

    call_price = (
        alpha * (S ** beta)
        - alpha * _phi(S, T, beta, I, I)
        + _phi(S, T, 1.0, I, I)
        - _phi(S, T, 1.0, K, I)
        - K * _phi(S, T, 0.0, I, I)
        + K * _phi(S, T, 0.0, K, I)
    )
    euro_price = black_scholes_price(S, K, T, r, sigma, q, otype=1.0)
    return max(call_price, euro_price, max(0.0, S - K))

def bjerksund_stensland_price(S, K, T, r, sigma, q=0.0, otype=1.0):
    """
    Calculate the Bjerksund-Stensland American option price for a range of parameters.
    otype: 1.0 for call, -1.0 for put
    """
    S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
    K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    otype_arr = np.atleast_1d(np.asarray(otype, dtype=np.float64))

    b_S, b_K, b_T, b_r, b_sigma, b_q, b_otype = np.broadcast_arrays(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr, otype_arr
    )

    prices = np.empty_like(b_S, dtype=np.float64)
    flat_S = b_S.ravel()
    flat_K = b_K.ravel()
    flat_T = b_T.ravel()
    flat_r = b_r.ravel()
    flat_sigma = b_sigma.ravel()
    flat_q = b_q.ravel()
    flat_otype = b_otype.ravel()
    flat_out = prices.ravel()

    for i in range(len(flat_S)):
        s_val, k_val = flat_S[i], flat_K[i]
        t_val = max(flat_T[i], 1e-5)
        r_val, q_val = flat_r[i], flat_q[i]
        sig_val = max(flat_sigma[i], 1e-5)

        if flat_otype[i] > 0:
            flat_out[i] = _bjerksund_stensland_call_scalar(s_val, k_val, t_val, r_val,q_val, sig_val)
        else:
            flat_out[i] = _bjerksund_stensland_call_scalar(k_val, s_val, t_val, q_val,r_val, sig_val)

    if np.isscalar(S) and np.isscalar(K) and np.isscalar(T) and np.isscalar(sigma):
        return float(prices[0])
    return prices