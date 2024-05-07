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
import sys 


sys.path.insert(1, '/home/digonto/Codes/Practical_Lattice_v2/jackknife_codes/')

import jackknife 

#we build the covariance between different states 
#we consider that all the files have same length
#we return the states with their errors as well
def covariance_between_states_L20(energy_cutoff):
    path_to_files = '/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/lattice_data/KKpi_interacting_spectrum/Lattice_data/KKpi_L20/L20_data_modified_for_covariance/'

    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    max_state_num = 10

    check_total_states = 0
    state_file_list = []

    for moms in list_of_mom:
        for states in range(max_state_num):
            filename = path_to_files + 'mass_' + moms + '_state' + str(states) + '.jack'
            #print("checking file = ",filename)
            if(os.path.exists(filename)):
                print("found file = ",filename)
                (temp0, datatemp) = np.genfromtxt(filename, unpack=True, skip_header=1)
                resampled_datatemp = jackknife.jackknife_resampling(datatemp)
                avgtemp = jackknife.jackknife_average(resampled_datatemp)
                if(avgtemp<energy_cutoff):
                    check_total_states = check_total_states + 1 
                    state_file_list.append(filename)
    

    covariance_matrix = np.zeros((check_total_states,check_total_states))

    states_avg = np.zeros((check_total_states))
    states_err = np.zeros((check_total_states)) 

    for i in range(check_total_states):
        for j in range(check_total_states):
            file1 = state_file_list[i]
            file2 = state_file_list[j]

            (temp0, data1) = np.genfromtxt(file1, unpack=True, skip_header=1)
            (temp0, data2) = np.genfromtxt(file2, unpack=True, skip_header=1) 

            resampled_data1 = jackknife.jackknife_resampling(data1)
            resampled_data2 = jackknife.jackknife_resampling(data2)

            avg1 = jackknife.jackknife_average(resampled_data1)
            avg2 = jackknife.jackknife_average(resampled_data2)

            err1 = jackknife.jackknife_error(resampled_data1)
            err2 = jackknife.jackknife_error(resampled_data2) 

            size = len(resampled_data1)
            n = size 

            sum_res = 0.0
            for k in range(size):
                sum_res = sum_res + ((resampled_data1[k] - avg1)/err1)*((resampled_data2[k] - avg2)/err2)

            covariance_matrix[i][j] =  ((n-1.0)/n)*sum_res 

        states_avg[i] = avg1 
        states_err[i] = err1   



    return states_avg, states_err, covariance_matrix 



st, sterr, cov = covariance_between_states_L20(0.38)

with np.printoptions(precision=6, suppress=True):
    for i in range(len(st)):
        print(st[i], sterr[i])
    
print("=======================")


for i in range(len(st)):
    for j in range(len(st)):
        with np.printoptions(precision=3, suppress=True):
            print("%.3f" %cov[i][j], end=' ')
            
    print('\n')
#    print(np.asmatrix(cov))
 
