# -*- coding: utf-8 -*-
"""
Created on STn Nov 24 20:20:33 2024

@aTthor: lambr
"""
import numpy as np
import matplotlib.pyplot as plt
from FEM import FEM_1D_nl

L=1.

lmbd = 0.001

n = 2

ex3=FEM_1D_nl((0, L), 50, interp='quadratic')

def f(theta):
    return -lmbd * theta**n

parameters = [1., 0., lambda theta: f(theta)]

ex3.dir_cond((None,1.))

ex3.neu_cond((0.,None))

ex3.eq_param(parameters)

max_it=100
epsilon=10**(-8)

ex3.soltest=np.random.rand(ex3.nnx)

#%% Part 1

lmbd=0.001

ex3.soltest=np.random.rand(ex3.nnx)
    
ex3.solve(max_it=max_it, epsilon=epsilon)

plt.plot(ex3.nodes, ex3.solution, label=f'λ = {round(lmbd,3)}')
plt.ticklabel_format(axis='y', style='plain', useOffset=False)
plt.title(f'Solution for n={n} and λ={lmbd}')
plt.xlabel('x')
plt.ylabel('θ')
    
plt.show()

#%% Part 2

ex3.soltest=np.random.rand(ex3.nnx)


for lmbd in np.linspace(0.001, 0.5, 10):
    
    ex3.solve(max_it=max_it, epsilon=epsilon)
    
    plt.plot(ex3.nodes, ex3.solution, label=f'λ = {round(lmbd,3)}')
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('θ')


plt.title(f'Solutions for n={n}')    
plt.show()

#%% Part 3

lmbd=0.001
dl=0.002
min_dl=0.001
n=-2

ex3.soltest=np.random.rand(ex3.nnx)

sols_st={}

while dl>min_dl and lmbd<=0.5:
    
    try:
        
        ex3.solve(max_it=max_it, epsilon=epsilon)
        
        plt.plot(ex3.nodes, ex3.solution, label=f'λ = {round(lmbd,3)}')
            
        sols_st[lmbd]=ex3.solution
        
        
        #dl += 0.2*dl
        
        lmbd+=dl   
    except:
        
        lmbd-=dl
        dl -= 0.2*dl
        lmbd+=dl
        
if lmbd<0.5:
    print(f'\n Not converging after λ={lmbd-dl} \n')

plt.ticklabel_format(axis='y', style='plain', useOffset=False)
plt.legend()
plt.xlabel('x')
plt.ylabel('θ')
plt.title('Solutions in interval [0.001 , 0.5] but at 0.35 stops converging')
plt.show()

#%% Part 4

lmbd=0.001
dl=0.002
min_dl=0.001
n=-2
max_it=100

with open('./Assignment 3/solution.txt') as file:
    sol_txt=np.loadtxt(file)

num_el_quad = int((len(sol_txt)-1)/2)
ex3=FEM_1D_nl((0, L), num_el_quad, interp='quadratic')

# num_el_lin = int(len(sol_txt)-1)
# ex3=FEM_1D_nl((0, L), num_el_lin, interp='linear')

sols_unst={}

def f(theta, par):
    return -par * theta**n

parameters = [1., 0., f]

ex3.dir_cond((None,1.))

ex3.neu_cond((0.,None))

ex3.eq_param(parameters)

ex3.soltest = sol_txt[:,1]

while dl>min_dl and lmbd<=0.5:
    print(lmbd)
    try:
        
        print(f'trying with {lmbd}')
        ex3.solve_par_cont(lmbd, max_it=max_it, epsilon=epsilon)
        
        plt.plot(ex3.nodes, ex3.solution, label=f'λ = {round(lmbd,3)}')
        
        sols_unst[lmbd]=ex3.solution
        
        #dl += 0.1*dl
        
        print(f'converged with {lmbd}')
        
        lmbd+=dl
        
        du_dp=np.linalg.solve(ex3.Jac, -ex3.dR_dpar)
        ex3.soltest=np.copy(ex3.solution)+dl*du_dp
        
        print(f'next is {lmbd}')
    except:
        
        lmbd-=dl
        dl -= 0.2*dl
        lmbd+=dl
    

if lmbd<0.5:
    print(f'\n Not converging after λ={lmbd-dl} \n')

plt.ticklabel_format(axis='y', style='plain', useOffset=False)
plt.legend()
plt.xlabel('x')
plt.ylabel('θ')
plt.title('Parametric continuation with n=-2  with in. sol. from solution.txt')
plt.show()

#%% Part 4 - Comparison

for l in sols_st.keys():
    plt.scatter(ex3.nodes, sols_st[l], color='blue')
    plt.scatter(ex3.nodes, sols_unst[l], color='red')
    plt.title(f'solutions for λ={round(l,4)}')
    plt.show()
    
for l in sols_st.keys():
    plt.plot(ex3.nodes, sols_st[l]-sols_unst[l], color='orange')
    plt.title(f'sols1 - soltxt for λ={round(l,4)}')
    plt.show()