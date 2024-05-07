
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyPDF2 import PdfMerger

import os 
import subprocess 
import math 

#This does jackknife resampling on a dataset 'data'
#'data' is a 1d vec 
def jackknife_resampling(data):
    n = len(data)
    resampled_data = []
    for i in range(n):
        sum_res = 0.0 
        for j in range(n):
            if(i!=j):
                sum_res = sum_res + data[j]
            
        resampled_data.append(sum_res/(n-1.0))
    
    np_resampled_data = np.array(resampled_data)

    return np_resampled_data 


#This gives the average based on the jackknife resampled data 
def jackknife_average(resampled_data):
    n = len(resampled_data)

    sum_res = 0.0
    for i in range(n):
        sum_res = sum_res + resampled_data[i]

    
    avg = sum_res/n 

    return avg 

#This gives the error based on the jackknife resampled data 
def jackknife_error(resampled_data):
    n = len(resampled_data)

    avg = jackknife_average(resampled_data)

    sum_res = 0.0
    for i in range(n):
        sum_res = sum_res + (resampled_data[i] - avg)**2 

    return np.sqrt(((n-1.0)/n)*sum_res) 

