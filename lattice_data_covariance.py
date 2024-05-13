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

def E_to_Ecm(En, P):
    return np.sqrt(En**2 - P**2)

def Esq_to_Ecmsq(En, P):
    return En**2 - P**2

def Ecmsq_to_Esq(Ecm, P):
    return Ecm**2 + P**2

#we build the covariance between different states 
#we consider that all the files have same length
#we return the states with their errors as well
def covariance_between_states_L20(energy_cutoff, list_of_mom):
    path_to_files = '/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/lattice_data/KKpi_interacting_spectrum/Lattice_data/KKpi_L20/L20_data_modified_for_covariance/'

    #list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    max_state_num = 10

    check_total_states = 0
    state_file_list = []
    nP_list = []

    xi = 3.444
    L = 20
    Lbyas = L*xi 
    pi = math.acos(-1.0) 
    twopibyLbyas = 2.0*pi/Lbyas 

    for moms in list_of_mom:
        for states in range(max_state_num):
            filename = path_to_files + 'mass_' + moms + '_state' + str(states) + '.jack'
            #print("checking file = ",filename)
            if(os.path.exists(filename)):
                print("found file = ",filename)
                (temp0, datatemp) = np.genfromtxt(filename, unpack=True, skip_header=1)
                resampled_datatemp = jackknife.jackknife_resampling(datatemp)
                avgtemp = jackknife.jackknife_average(resampled_datatemp)
                if(moms=='000_A1m'):
                    nPx = 0
                    nPy = 0
                    nPz = 0
                    #nP_list.append([nPx,nPy,nPz])
                elif(moms=='100_A2'):
                    nPx = 1
                    nPy = 0
                    nPz = 0
                    #nP_list.append([nPx,nPy,nPz])
                elif(moms=='110_A2'):
                    nPx = 1
                    nPy = 1
                    nPz = 0
                    #nP_list.append([nPx,nPy,nPz])
                elif(moms=='111_A2'):
                    nPx = 1
                    nPy = 1
                    nPz = 1
                    #nP_list.append([nPx,nPy,nPz])
                elif(moms=='200_A2'):
                    nPx = 2
                    nPy = 0
                    nPz = 0
                    #nP_list.append([nPx,nPy,nPz])  
                Px = twopibyLbyas*nPx 
                Py = twopibyLbyas*nPy 
                Pz = twopibyLbyas*nPz 
                P = np.sqrt(Px*Px + Py*Py + Pz*Pz)
                avgtemp_Ecm = E_to_Ecm(avgtemp,P)  
                
                if(avgtemp_Ecm<energy_cutoff):
                    check_total_states = check_total_states + 1 
                    state_file_list.append(filename)
                    nP_list.append([nPx,nPy,nPz])
                    
    
    state_no = []
    temp_state_no = 0
    for i in range(len(state_file_list)):
        if(i==0):
            state_no.append(temp_state_no)
            temp_nPx = nP_list[i][0]
            temp_nPy = nP_list[i][1]
            temp_nPz = nP_list[i][2]
            temp_state_no = temp_state_no + 1 
        else:
            current_nPx = nP_list[i][0]
            current_nPy = nP_list[i][1]
            current_nPz = nP_list[i][2]
            if(temp_nPx==current_nPx and temp_nPy==current_nPy and temp_nPz==current_nPz):
                state_no.append(temp_state_no)
                temp_state_no = temp_state_no + 1
            else:
                temp_state_no = 0
                state_no.append(temp_state_no)
                temp_state_no = temp_state_no + 1 
            
            temp_nPx = current_nPx
            temp_nPy = current_nPy
            temp_nPz = current_nPz

    covariance_matrix = np.zeros((check_total_states,check_total_states))

    states_avg = np.zeros((check_total_states))
    states_err = np.zeros((check_total_states)) 

    for i in range(check_total_states):
        for j in range(check_total_states):
            file1 = state_file_list[i]
            file2 = state_file_list[j]

            (temp0, data1) = np.genfromtxt(file1, unpack=True, skip_header=1)
            (temp0, data2) = np.genfromtxt(file2, unpack=True, skip_header=1) 

            nPx1 = nP_list[i][0]
            nPy1 = nP_list[i][1]
            nPz1 = nP_list[i][2]
            Px1 = twopibyLbyas*nPx1 
            Py1 = twopibyLbyas*nPy1 
            Pz1 = twopibyLbyas*nPz1 
            P1 = np.sqrt(Px1*Px1 + Py1*Py1 + Pz1*Pz1)
            
            nPx2 = nP_list[j][0]
            nPy2 = nP_list[j][1]
            nPz2 = nP_list[j][2]
            Px2 = twopibyLbyas*nPx2 
            Py2 = twopibyLbyas*nPy2 
            Pz2 = twopibyLbyas*nPz2 
            P2 = np.sqrt(Px2*Px2 + Py2*Py2 + Pz2*Pz2)

            resampled_data1 = jackknife.jackknife_resampling(data1)
            resampled_data2 = jackknife.jackknife_resampling(data2)

            resampled_data1_ecm = np.zeros((len(resampled_data1)))
            resampled_data2_ecm = np.zeros((len(resampled_data2)))

            for k in range(len(resampled_data1)):
                resampled_data1_ecm[k] = E_to_Ecm(resampled_data1[k],P1)
            for l in range(len(resampled_data2)):
                resampled_data2_ecm[l] = E_to_Ecm(resampled_data1[l],P1)
            
            avg1 = jackknife.jackknife_average(resampled_data1_ecm)
            avg2 = jackknife.jackknife_average(resampled_data2_ecm)

            err1 = jackknife.jackknife_error(resampled_data1_ecm)
            err2 = jackknife.jackknife_error(resampled_data2_ecm) 

            size = len(resampled_data1)
            n = size 

            sum_res = 0.0
            for k in range(size):
                sum_res = sum_res + ((resampled_data1[k] - avg1)/err1)*((resampled_data2[k] - avg2)/err2)

            covariance_matrix[i][j] =  ((n-1.0)/n)*sum_res 

        

        
        states_avg[i] = avg1 
        states_err[i] = err1   


    np_state_no = np.array(state_no)

    return states_avg, states_err, nP_list, np_state_no, covariance_matrix 


'''
st, sterr, nP_list,state_no, cov = covariance_between_states_L20(0.38)

with np.printoptions(precision=6, suppress=True):
    for i in range(len(st)):
        print(st[i], sterr[i], nP_list[i], state_no[i])
    
print("=======================")


for i in range(len(st)):
    for j in range(len(st)):
        with np.printoptions(precision=3, suppress=True):
            print("%.3f" %cov[i][j], end=' ')
            
    print('\n')
#    print(np.asmatrix(cov))
 
for i in range(5):
    s = np.linspace(0,1,10)

print(s)
'''