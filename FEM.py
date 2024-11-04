# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

class FEM_1D:
    
    def __init__(self, domain, nex, interp='linear'):
        
        self.interp=interp
        self.nn_el=2
        self.domain=domain
        self.nex=nex
        self.nnx=(self.nn_el-1)*self.nex+1
        self.nodes=np.linspace(domain[0], domain[1], num=(nex*self.nn_el-1))
        self.elements=[]
        for i in range(self.nex):
            self.elements.append((i,i+1))
        self.dirbc=[]
        self.neubc=[]
        
    def eq_param(self, parameters):
        
        self.par=parameters          # u0 , u , du/dx , d2u/dx2
    
    def nop(self, ln, el):
        
        return self.elements[el][ln]
    
    def dir_node(self, node, value):
        
        self.dirbc.append([node, value])
    
    
    def plotmesh(self):

        plt.scatter(self.nodes, np.zeros(len(self.nodes)), color='blue')
        
        
        if len(self.dirbc)>0:
            plt.scatter(self.nodes[self.dirbc[:][0]][0], np.zeros(len(self.dirbc)), color='red')
        plt.show()
    
    def ph(self, x):
        
        return np.array([1.-x , x])
    
    def phd(self):
        
        return np.array([-1., 1.])
    
    def abfind(self, nel):
        
        wgp=1.
        gp=0.5
        
        x=0.
        x1=0.
        
        ph = self.ph(gp)
        phd = self.phd()
        
        for n in range(self.nn_el):
            
            x=x+self.nodes[self.nop(n,nel)]*ph[n]
            x1=x1+self.nodes[self.nop(n,nel)]*phd[n] # dx / dξ
        
        phx = phd/x1
        
        a = self.par[1]
        #a2 = self.par[3]
        
        for m in range(self.nn_el):
            
            m1=self.nop(m, nel)
            
            for n in range(self.nn_el):
                
                n1=self.nop(n, nel)
                                
                self.A[m1,n1]=(self.A[m1,n1]
                               -wgp*x1*phx[m]*phx[n]
                               +a*wgp*x1*ph[m]*ph[n])
            
    
    def axb(self):
        
        self.A=np.zeros((self.nnx,self.nnx))
        self.b=np.zeros(self.nnx)
        
        for nel in range(len(self.elements)):
            
            self.abfind(nel)
        
        # BC
        
        for node, value in self.dirbc:
            
            self.A[node]=0.
            
            self.A[node, node]=1.
            
            self.b[node]=value
        
        
    
class FEM_2D:
    
     
    def __init__(self, domain, ne, interp='linear'):
        
        self.nn_el=4
        self.domain=domain
        self.xdomain=domain[0]
        self.ydomain=domain[1]
        self.nex=ne[0]
        self.ney=ne[1]
        self.nnx=self.nex+1
        self.nny=self.ney+1     
        self.xnodes=np.linspace(self.xdomain[0], self.xdomain[1], num=self.nnx)
        self.ynodes=np.linspace(self.ydomain[0], self.ydomain[1], num=self.nny)
        self.nodes=np.meshgrid(self.xnodes, self.ynodes)
        self.nodes=np.moveaxis(np.array(self.nodes), 0, self.nodes[0].ndim).reshape(-1, len(self.nodes))
        self.nodes=self.nodes[self.nodes[:,1].argsort(kind='mergesort')]
        self.nodes=self.nodes[self.nodes[:,0].argsort(kind='mergesort')]
        self.elements=[]
        for i in range(self.nex):
            for j in range(self.ney):
                self.elements.append((i*self.nny+j,
                                      i*self.nny+j+1,
                                      (i+1)*self.nny+j,
                                      (i+1)*self.nny+j+1))
        self.dirbc=[]
        self.neubc=[]
    
             
    def nop(self, ln, el):
        
        return self.elements[el][ln]
        
    def dir_node(self, node, value):
        
        self.dirbc.append([node, value])
        
    
    def plotmesh(self):
    
        plt.scatter(self.nodes[:,0], self.nodes[:,1])
        
        if len(self.dirbc)>0:
            plt.scatter(self.nodes[np.array(self.dirbc)[:,0],0], self.nodes[np.array(self.dirbc)[:,0],1], color='red')
        plt.show()
  

        
test=FEM_1D((0,10), 6)

test.plotmesh()

# test2=FEM_2D(((0,10),(0,5)), (11,6))

# test2.plotmesh()


# test.dir_node(0, 0)

# test.plotmesh()

# test2.dir_node(11, 0)
# test2.dir_node(12, 0)

# test2.plotmesh()

a=-1.0
parameters = [0, a, 0, 1]  # u0 , u , du/dx , d2u/dx2

test.dir_node(0, 1)

test.eq_param(parameters)
test.axb()