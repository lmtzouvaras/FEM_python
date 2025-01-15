# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:33:46 2024

@author: lambr
"""
import numpy as np
import matplotlib.pyplot as plt
from FEM import FEM_2D

T0=20
T1=100
h=5

for theta in [np.pi/2, np.pi/3, np.pi/6]:

    ex2 = FEM_2D(((0,6),(0,3)), (12,6), interp='quadratic')
    
    ex2.eq_param([0., 0., lambda r,z : 1./r, 1., lambda r,z : 0., 1.])           # constant , T , dT/dr , d2T/dr2, dT/dz, d2T/dz2
    
    #ex2.plotmesh()
    
    ex2.dir_cond((0.,None), T1)
    
    ex2.rob_cond((None,3.), [h, -h*T0, 'n_in'])
    
    ex2.rob_cond((None,0.), [h, -h*T0, 'n_in'])
    
    ex2.neu_cond((6.,None), [0., 'n_out'])
    
    #ex2.plotmesh()
    
    #ex2.plotmesh(show_elements=True)
    
    ub=ex2.xdomain[1]*np.array([0., 1.])-np.array([0., 3./np.tan(theta)])
    
    ex2.to_trapezoid(ub)
    
    ex2.move([1., 0.])
        
    #ex2.plotmesh(show_elements=True, grid=False)
    
    T = ex2.solve()
    
    Tmat = np.reshape(T, (ex2.nnx, ex2.nny))
    Tmat=Tmat.transpose()

    ex2.plot_cont('r', 'z', 'T', 'Temperature distribution ( C )', 100)