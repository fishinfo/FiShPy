#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:43:44 2019

@author: fguignar1
"""
import numpy as np
import scipy.integrate
import warnings

def nsrk(data, log_trsf = False):
    '''
    Normal scale rule for kernel density estimation
    -----------------------------------------------
    
    Bandwidth selector for non-parametric estimation. Estimates the optimal 
    AMISE bandwidth using the normal scale rule with Gaussian kernel.    

    Input : - data: 1-D np.array ; Univariate data.
            - log_trsf: boolean; If True, the data are log-transformed (usually 
                                 used for skewed positive data), False by default. 
 
    Output : - bandwidth value
    
    References
    ----------
    M.P. Wand and M.C. Jones (1995). Kernel Smoothing, Chapman and Hall, London.
    
    '''
    if log_trsf == True :
        data = np.log(data)
    elif log_trsf == False :
        None
    else :
        raise Exception('log_trsf should be a boolean')

    n = data.shape[0]
    stdev = data.std(ddof=1)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = (q75 - q25) /1.349
    scalest = min(stdev, iqr)
    output = scalest*(4/(3*n))**(1/5)
    return output

def _linbin(data, gpoints, trunc = True):
    '''
    Internal function.
    Linear binning. Translated from KernSmooth R package. 
    '''
    n = data.shape[0]
    M = gpoints.shape[0]
    a = gpoints[0]
    b = gpoints[-1]
    
    # initilisaion of gcounts:
    gcounts = np.zeros(M)
    Delta = (b-a)/(M-1)
    
    for i in range(n):
        lxi = ((data[i]-a)/Delta)+1
        li = int(lxi)
        
        rem = lxi - li
        
        if (li >= 1) and (li < M) :
            gcounts[li-1] = gcounts[li-1] + 1-rem
            gcounts[li] = gcounts[li] + rem
            
        elif (li < 1) and (trunc == False):
            gcounts[0] = gcounts[0] + 1
            
        elif (li >= M) and (trunc == False):
            gcounts[M-1] = gcounts[M-1] + 1
            
    return gcounts
            

def _binned_bkfe(gcounts, drv, h, a, b, trunc = True):
    '''
    Internal function.
    Translated from KernSmooth R package, part of 'bkfe' function. 
    
    Input : - gcounts = 1-D np.array,
            - drv = Functional derivative, float,
            - h = Bandwidth value for the density function of the data, float,
            - a = Range minimum, float,
            - b = Range maximum, float,
            - trunc = Boolean,
 
    Output : - Estimate of psi hat
    '''
    resol = gcounts.shape[0]
    
    ## Set the sample size and bin width    
    n = gcounts.sum()
    delta = (b-a)/(resol-1)

    ## Obtain kernel weights    
    tau = drv+4
    L = min(int(tau*h/delta), resol)
    if L == 0:
        warnings.warn("WARNING : Binning grid too coarse for current (small) bandwidth: consider increasing 'resolution'")
    lvec = np.arange(L+1)
    arg = lvec * delta / h
    
    dnorm = lambda x : np.exp(-np.square(x)/2) / np.sqrt(2*np.pi)
    kappam = dnorm(arg) / h**(drv+1)
    hmold0 = 1
    hmold1 = arg
    hmnew = 1
    if drv >= 2 :
        for i in np.arange(2, drv+1):
            hmnew = arg*hmold1 - (i-1)*hmold0
            hmold0 = hmold1         # Compute mth degree Hermite polynomial
            hmold1 = hmnew          # by recurrence.
    kappam = hmnew * kappam
            
    ## Now combine weights and counts to obtain estimate
    P = 2**(int(np.log(resol+L+1)/np.log(2))+1)
    kappam = np.concatenate((kappam, np.zeros(P-2*L-1), kappam[1:][::-1]), axis = 0)
    Gcounts = np.concatenate((gcounts, np.zeros(P-resol)), axis = 0)
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)
    
    gcounter = gcounts * (np.real(np.fft.ifft(kappam*Gcounts)))[0:resol]
    gcounter = gcounter.sum()
    
    return gcounter/n**2

def dpik(data, log_trsf = False, resol = 401, trunc = True):
    '''
    Direct plug-in method for kernel density estimation
    ---------------------------------------------------
    
    Bandwidth selector for non-parametric estimation. Estimates the optimal AMISE 
    bandwidth using the direct plug-in method with 2 levels for the Parzen-Rosenblatt 
    estimator with Gaussian kernel.
    Translated code of a part of 'dpik' function from the KernSmooth R package of 
    Wand and Ripley.

    Input : - data: 1-D np.array ; Univariate data.
            - log_trsf: boolean; If True, the data are log-transformed (usually 
                                 used for skewed positive data), False by default. 
            - resol: float; Number of equally-spaced points (as defined in 
                            KernSmooth R package)
            - trunc: boolean; Range of data to be ignored (as defined in 
                              KernSmooth R package)
 
    Output : - bandwidth value

    References
    ----------
    M.P. Wand and M.C. Jones (1995). Kernel Smoothing, Chapman and Hall, London.
    
    '''
    if log_trsf == True :
        data = np.log(data)
    elif log_trsf == False :
        None
    else :
        raise Exception('log_trsf should be a boolean')

    data_min = data.min()
    data_max = data.max()
    n = data.shape[0]
    stdev = data.std(ddof=1)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = (q75 - q25) /1.349
    scalest = min(stdev, iqr)
    data_scaled = (data - data.mean())/scalest
    min_scaled = (data_min - data.mean())/scalest
    max_scaled = (data_max - data.mean())/scalest
    gpoints = np.linspace(min_scaled, max_scaled, resol)
    gcounts = _linbin(data_scaled, gpoints, trunc)
    delta_0 = 1/((4*np.pi)**(1/10))
    alpha = (2*np.sqrt(2)**9/(7*n))**(1/9)   
    psi6hat = _binned_bkfe(gcounts, 6, alpha, min_scaled, max_scaled)
    alpha = (-3*np.sqrt(2/np.pi)/(psi6hat*n))**(1/7)
    psi4hat = _binned_bkfe(gcounts, 4, alpha, min_scaled, max_scaled)
    output = (scalest * delta_0 * (1/(psi4hat * n))**(1/5))
    return output

def SEP_FIM(data, h, log_trsf = False, resol = 1000, tol = 'default'):
    '''
    Fisher-Shannon method
    ---------------------
    
    Non-parametric estimates of the Shannon Entropy Power (SEP),
    the Fisher Information Measure (FIM), and the Fisher-Shannon Complexity (FSC)
    using kernel density estimators with Gaussian kernel.
             
    Input : - data: 1-D np.array ; Univariate data.
            - h: float; Bandwidth value.
            - log_trsf: boolean; If True, the data are log-transformed (usually 
                                 used for skewed positive data), False by default.
            - resol : integer; Number of equally-spaced points over which function
                               approximations are computed and integrated.
            - tol : float; Tolerance to avoid dividing by zero values. 
                           The machine epsilon is used by default.
 
    Output : - Shannon Entropy Power (SEP)
             - Fisher Information Measure (FIM)
             - Fisher-Shannon Complexity (FSC)
             
    Notes
    -----
    This Python code was developed and used for the following papers:
    F. Guignard, M. Laib, F. Amato and M. Kanevski (in prep). Advanced analysis of
    temporal data using Fisher-Shannon information : theoretical development and 
    application to geoscience.
    
    References
    ----------
    F. Guignard, M. Laib, F. Amato and M. Kanevski (2020). Advanced analysis of
    temporal data using Fisher-Shannon information : theoretical development and 
    application in geoscience, Frontiers in Earth Science, 8:255.
    
    C. Vignat, J.F Bercher (2003). Analysis of signals in the Fisher–Shannon 
    information plane, Physics Letters A, 312, 190, 27 – 33.
    
    Example
    --------
    import FiShPy.FiSh as FS
    import numpy as np
    data = np.random.normal(size = 1000)
    h = FS.dpik(data)
    SEP, FIM, FSC = FS.SEP_FIM(data, h)

    '''
    
    
    
    if tol == 'default':
        tol = np.finfo(float).eps
    elif type(tol) == float:
        None
    else:
        raise Exception('tol should be a float')

    n = data.shape[0]
    integ_start = data.min()
    integ_end = data.max()

    Accu_f = np.zeros(resol)
    Accu_f_drv = np.zeros(resol) 

    if log_trsf == False :

        x_grid = np.linspace(integ_start, integ_end, resol) 
        
        for i in range(n) :
            dist = x_grid - np.repeat(data[i], resol)
            kern = np.exp(-dist**2/ (2 * h**2))
            Accu_f += kern
            Accu_f_drv += dist * kern

        np.seterr(divide='ignore', invalid='ignore')
        FIM = np.where(Accu_f>tol, np.square(Accu_f_drv) / Accu_f, 0)
        np.seterr(divide='warn', invalid='warn')           
        FIM = scipy.integrate.simps(FIM, x=x_grid)       
        FIM *=  1 / (np.sqrt(2*np.pi) * h**5 * n)
        
        f = Accu_f / (np.sqrt(2*np.pi) * h * n)
        np.seterr(divide='ignore', invalid='ignore')
        H = np.where(f>tol, -f * np.log(f), 0)  
        np.seterr(divide='warn', invalid='warn')
        H = scipy.integrate.simps(H, x=x_grid)

    elif log_trsf == True :

        y = np.log(data)
        y_grid = np.linspace(np.log(integ_start), np.log(integ_end), resol)
        
        for i in range(n) :
            dist = y_grid - np.repeat(y[i], resol)
            kern =  np.exp(-dist**2/ (2 * h**2))
            Accu_f += kern
            Accu_f_drv += dist * kern
    
        back_x_grid = np.exp(y_grid)
        Accu_f = Accu_f/h
        Accu_f_drv = Accu_f_drv/h**3 + Accu_f
        
        np.seterr(divide='ignore', invalid='ignore')
        FIM = np.where(Accu_f>tol, np.square(Accu_f_drv) / Accu_f, 0) ####
        np.seterr(divide='warn', invalid='warn')
        FIM = FIM / back_x_grid**3
        FIM = scipy.integrate.simps(FIM, x=back_x_grid)
        FIM *=  1 / (np.sqrt(2*np.pi) * n)

        f = Accu_f / (np.sqrt(2*np.pi) * n * back_x_grid)
        np.seterr(divide='ignore', invalid='ignore')
        H = np.where(f>tol, -f * np.log(f), 0)
        np.seterr(divide='warn', invalid='warn')
        H = scipy.integrate.simps(H, x=back_x_grid)

    else :
        raise Exception('log_trsf should be a boolean')
        
    SEP = np.exp(2*H)/ (2*np.pi * np.exp(1))
    FSC = SEP * FIM
    
    if FSC<1:
        warnings.warn("WARNING : FSC < 1. The problem could be related to kernel density estimation, bad bandwidth selection, or there are not enough points.")
 
    return SEP, FIM, FSC