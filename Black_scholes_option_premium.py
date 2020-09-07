#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
init_printing()


# In[7]:


def black_scholes_call_div(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #q: dividend yield
    #sigma: Implied volatility
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call


# In[3]:


def black_scholes_put_div(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #q: dividend yield 
    #sigma: Implied volatility
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put


# In[14]:


def euro_vanilla_dividend(S, K, T, r, q, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #q: dividend yield
    #sigma: Implied volatility
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result


# In[15]:


def black_scholes_call_div_sym(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #q: dividend yield 
    #sigma: Implied volatility
    
    N = Normal('x', 0.0, 1.0)
    
    d1 = (sy.ln(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    
    call = S * sy.exp(-q * T) * cdf(N)(d1) - K * sy.exp(-r * T) * cdf(N)(d2)
    
    return call


# In[16]:


def black_scholes_call_put_sym(S, K, T, r, q, sigma):

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #q: dividend yield 
    #sigma: Implied volatility
    
    N = Normal('x', 0.0, 1.0)
    
    d1 = (sy.ln(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    
    put = K * sy.exp(-r * T) * cdf(N)(-d2) - S * sy.exp(-q * T) * cdf(N)(-d1)
    
    return put


# In[17]:


def sym_euro_vanilla_dividend(S, K, T, r, q, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #q: dividend yield
    #sigma: Implied volatility
    
    N = Normal('x', 0.0, 1.0)
    
    d1 = (sy.ln(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    
    if option == 'call':
        result = S * sy.exp(-q * T) * cdf(N)(d1) - K * sy.exp(-r * T) * cdf(N)(d2)
    if option == 'put':
        result = K * sy.exp(-r * T) * cdf(N)(-d2) - S * sy.exp(-q * T) * cdf(N)(-d1)
        
    return result


# In[18]:


sym_euro_vanilla_dividend(100, 110, 2, 4, 1, 10, option = 'call')


# In[ ]:




