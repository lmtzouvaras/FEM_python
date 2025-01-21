# -*- coding: utf-8 -*-
"""
Created on STn Nov 24 20:20:33 2024

@aTthor: lambr
"""
import numpy as np
import matplotlib.pyplot as plt
from FEM import FEM_1D

h=100
k=398.
P=8.
Ac=4.
L=100.
Too=25.

allsolsA=[]
allsolsB=[]
qfA=[]
qfB=[]

for n_elements in [10, 20, 100, 200, 400, 800]:
    for interpolation in ['linear', 'quadratic']:
        
        ex1=FEM_1D((0, L), n_elements, interp=interpolation)
        
        c = 10**(-3) * Too*h*P/(k*Ac)
        a0 = - 10**(-3) * h*P/(k*Ac)
        a1 = 0.
        a2 = 1.
        
        parameters = [c, a0, a1, a2]
        
        # Part A
        print(f'Exercise 1 \n Part A \n {n_elements} elements, {interpolation} \n')
        
        ex1.dir_cond((100,30))
        
        ex1.eq_param(parameters)
        
        T = ex1.solve()
        allsolsA.append(T)
        
        ex1.plotmesh()
        
        plt.plot(ex1.nodes, T)
        plt.title(f'Temperature distribution (A), {n_elements} elements, {interpolation}')
        plt.xlabel('x (mm)')
        plt.ylabel('T ($^\circ$C)')
        plt.show()
            
        qf=ex1.dsol_dx()*(-k*Ac)/1000
        qfA.append(qf[0])
            
        plt.plot(ex1.nodes, qf, color='orange')
        plt.title(f'Heating Rate distribution (A), {n_elements} elements, {interpolation}')
        plt.xlabel('x (mm)')
        plt.ylabel('qf (W)')
        plt.show()
        
        print('qf (x=0) = ', round(qf[0],6),' (W)')
        
        # Part B
        print(f'\n Part B \n {n_elements} elements, {interpolation} \n')
        
        ex1.reset()
        
        ex1.dir_cond((100,None))
        ex1.rob_cond((None, ( 10**(-3) * ( - h/k ) , 10**(-3) * ( h*Too/k ) )))
        
        T = ex1.solve()
        allsolsB.append(T)
        
        ex1.plotmesh()
        
        plt.plot(ex1.nodes, T)
        plt.title(f'Temperature distribution (B), {n_elements} elements, {interpolation}')
        plt.xlabel('x (mm)')
        plt.ylabel('T ($^\circ$C)')
        plt.show()
        
        qf=ex1.dsol_dx()*(-k*Ac)/1000
        qfB.append(qf[0])
                
        plt.plot(ex1.nodes, qf, color='orange')
        plt.title(f'Heating Rate distribution (B), {n_elements} elements, {interpolation}')
        plt.xlabel('x (mm)')
        plt.ylabel('qf (W)')
        plt.show()
        
        print('qf (x=0) = ', round(qf[0],6),' (W)')
        #print(h*(T[-1]-Too), 10**6 *qf[-1]/Ac)

plt.plot([10, 20, 100, 200, 400, 800], qfA[0::2], label='linear')
plt.plot([10, 20, 100, 200, 400, 800], qfA[1::2], label='quadratic')
plt.xlabel('number of Elements')
plt.ylabel('qf (x=0)')
plt.title('qf (x=0) - #elements (A)')
plt.legend()
#plt.yscale('log')
plt.show()

plt.plot([10, 20, 100, 200, 400, 800], qfB[0::2], label='linear')
plt.plot([10, 20, 100, 200, 400, 800], qfB[1::2], label='quadratic')
plt.xlabel('number of Elements')
plt.ylabel('qf (x=0)')
plt.title('qf (x=0) - #elements (B)')
plt.legend()
#plt.yscale('log')
plt.show()


RMS_A=[]
RMS_B=[]

for i in range(1,len(allsolsA)):
    
    if len(allsolsA[i])==len(allsolsA[i-1]):
           dif = np.array(allsolsA[i]-allsolsA[i-1])
           rms = np.sqrt(np.mean(dif**2))
           
           RMS_A.append(rms)
           
for i in range(1,len(allsolsB)):
    
    if len(allsolsB[i])==len(allsolsB[i-1]):
           dif = np.array(allsolsB[i]-allsolsB[i-1])
           rms = np.sqrt(np.mean(dif**2))
           
           RMS_B.append(np.format_float_scientific(np.float32(rms),5))
           

           