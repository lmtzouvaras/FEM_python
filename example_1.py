# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:22:53 2024

@author: lambr
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

ex1=FEM_1D((0, L), 1000, interp='quadratic')

c = 10**(-3) * Too*h*P/(k*Ac)
a0 = - 10**(-3) * h*P/(k*Ac)
a1 = 0.
a2 = 1.

parameters = [c, a0, a1, a2]    # constant , u , du/dx , d2u/dx2

# Part A

ex1.dir_cond((100,30))

ex1.eq_param(parameters)

T = ex1.solve()

plt.plot(ex1.nodes, T)
plt.title('Temperature distribution (A)')
plt.xlabel('x (mm)')
plt.ylabel('T ($^\circ$C)')
plt.show()
    
qf=ex1.dsol_dx()*(-k*Ac)/1000
    
plt.plot(ex1.nodes, qf, color='orange')
plt.title('Heating Rate distribution (A)')
plt.xlabel('x (mm)')
plt.ylabel('qf (W)')
plt.show()

print('qf (x=0) = ', qf[0],' (W)')

# Part B

ex1.reset()

ex1.dir_cond((100,None))
ex1.rob_cond((None, ( 10**(-3) * ( - h/k ) , 10**(-3) * ( h*Too/k ) )))

T = ex1.solve()

plt.plot(ex1.nodes, T)
plt.title('Temperature distribution (B)')
plt.xlabel('x (mm)')
plt.ylabel('T ($^\circ$C)')
plt.show()

qf=ex1.dsol_dx()*(-k*Ac)/1000
        
plt.plot(ex1.nodes, qf, color='orange')
plt.title('Heating Rate distribution (B)')
plt.xlabel('x (mm)')
plt.ylabel('qf (W)')
plt.show()