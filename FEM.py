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
        
        if interp=='linear':
            self.nn_el=2
        elif interp=='quadratic':
            self.nn_el=3
        elif interp=='cubic':
            self.nn_el=5

        self.domain=domain
        self.nex=nex
        self.nnx=(self.nn_el-1)*self.nex+1
        self.nodes=np.linspace(domain[0], domain[1], num=(self.nnx))
        self.elements=[]
        for i in range(self.nex):
            temp=[]
            for j in range(self.nn_el):
                temp.append(i*(self.nn_el-1)+j)              
            self.elements.append(tuple(temp))
        self.dirbc=[]
        self.neubc=[]
        self.robc=[]
        
    def eq_param(self, parameters):
        
        self.par=parameters          # u0 , u , du/dx , d2u/dx2
    
    def nop(self, ln, el):
        
        return self.elements[el][ln]
    
    def dir_node(self, node, value):
        
        self.dirbc.append([node, value])
        
    def dir_cond(self, values=(None,None)):
        
        if values[0]!=None:
            self.dirbc.append([0, values[0]])
        if values[-1]!=None:
            self.dirbc.append([self.nnx-1, values[-1]])
        
    def neu_cond(self, values=(None,None)):
        
        if values[0]!=None:
            self.neubc.append([0, values[0]])
        if values[-1]!=None:
            self.neubc.append([self.nnx-1, values[-1]])
            
    def rob_cond(self, values=(None, None)):
        
        if values[0]!=None:
            self.robc.append([0, values[0]])
        if values[-1]!=None:
            self.robc.append([self.nnx-1, values[-1]])
    
    
    def plotmesh(self):

        plt.scatter(self.nodes, np.zeros(len(self.nodes)), color='blue')
        
        
        if len(self.dirbc)>0:
            plt.scatter(self.nodes[np.array(self.dirbc)[:,0]], np.zeros(len(self.dirbc)), color='red')

        
        if len(self.neubc)>0:
            plt.scatter(self.nodes[np.array(self.neubc)[:,0]], np.zeros(len(self.neubc)), color='yellow')

        
        if len(self.robc)>0:
            for n, val in self.robc:
                plt.scatter(self.nodes[n], np.zeros(len(self.robc)), color='orange')
        
        
        
        plt.show()
    
    def ph(self, x):
        
        if self.interp=='linear':
            
            return np.array([1.-x , x])
        
        elif self.interp=='quadratic':
            
            return np.array([2.*x**2 -3.*x+1. ,
                             -4.*x**2 +4.*x ,
                             2.*x**2 -x])
        
        elif self.interp=='cubic':
            
            return np.array([2.*x**3 -3.*x**2 +1. ,
                             x**3 -2.*x**2 +x ,
                             -2.*x**3 +3.*x**2 ,
                             x**3 -x**2])
        
        
    
    def phd(self, x):
        
        if self.interp=='linear':
        
            return np.array([-1., 1.])
        
        elif self.interp=='quadratic':
            
            return np.array([4.*x -3. ,
                             -8.*x +4 ,
                             4.*x -1.])
        
        elif self.interp=='cubic':
            
            return np.array([6.*x**2 -6.*x ,
                             3.*x**2 -4.*x +1. ,
                             -6.*x**2 +6.*x ,
                             3.*x**2 -2.*x])
    
    def abfind(self, nel, ngp):
        
        if self.interp=='linear':
            
            self.ngp=1
                   
        elif self.interp=='quadratic':
            
            self.ngp=3
        
        elif self.interp=='cubic':
            
            self.ngp=5
        
        if ngp!=None:
            
            self.ngp=ngp
            
        if self.ngp==1:
            
            w=[1.]
            gp=[0.5]
            
        elif self.ngp==2:
            
            w=[0.5, 0.5]
            gp=[0.2115, 0.7885]
        
        elif self.ngp==3:
            
            w=[0.278, 0.4445, 0.278]
            gp=[0.1125, 0.5, 0.8875]
        
        elif self.ngp==4:
            
            w=[0.174, 0.326, 0.326, 0.174]
            gp=[0.0695, 0.33, 0.67, 0.9305]
        
        elif self.ngp==5:
            
            w=[0.1185, 0.2395, 0.2845, 0.2395, 0.1185]
            gp=[0.047, 0.231, 0.5, 0.769, 0.953]
        
        
        c=self.par[0]
        a0=self.par[1]
        a1=self.par[2]
        a2=self.par[3]
        
        #   loop over gauss points
        for p in range(len(gp)):
            
            x=0.
            x1=0.
            
            ph = self.ph(gp[p])
            phd = self.phd(gp[p])
            
            for n in range(self.nn_el):
                
                x=x+self.nodes[self.nop(n,nel)]*ph[n]
                x1=x1+self.nodes[self.nop(n,nel)]*phd[n] # dx / dξ
            
            phx = phd/x1
            
            for m in range(self.nn_el):
                
                m1=self.nop(m, nel)
                
                for n in range(self.nn_el):
                    
                    n1=self.nop(n, nel)
                    
                    self.A[m1,n1]=(self.A[m1,n1]
                                   +(a1-a2)*w[p]*x1*phx[m]*phx[n]
                                   +a0*w[p]*x1*ph[m]*ph[n])
                    
                self.b[m1]=(self.b[m1]
                               -c*w[p]*x1*ph[m])
        
        
    
    def axb(self, ngp=None):
        
        self.A=np.zeros((self.nnx,self.nnx))
        self.b=np.zeros(self.nnx)
        
        for nel in range(len(self.elements)):
            
            self.abfind(nel, ngp)
        
        
        c=self.par[0]
        a0=self.par[1]
        a1=self.par[2]
        a2=self.par[3]
        
        
        # Dirichlet BC
        
        for node, value in self.dirbc:
            
            self.A[node]=0.
            
            self.A[node, node]=1.
            
            self.b[node]=value
        
        # Neumann BC
        
        for node, value in self.neubc:
            
            if node==0:
                self.b[node]=self.b[node] + a2 * value
                
            if node==(self.nnx-1):
                self.b[node]=self.b[node] - a2 * value
                
        # Robin BC
        
        for node, values in self.robc:
            
            rob1, rob0 = values # Robin condition would be   du/dx = rob1 u + rob0
            
            if node==0:
                self.A[node,node]=self.A[node,node] - a2 * rob1
                self.b[node]=self.b[node] + a2 * rob0
                
            if node==(self.nnx-1):
                self.A[node,node]=self.A[node,node] + a2 * rob1
                self.b[node]=self.b[node] - a2 * rob0
            
        
        
            
    def solve(self, ngp=None):
        
        try:
            
            sol = np.linalg.solve(self.A, self.b)
            
        except:
            
            self.axb(ngp)
            
            sol = np.linalg.solve(self.A, self.b)
            
            
        return sol
    
    def reset(self):
        
        self.dirbc=[]
        self.neubc=[]
        self.robc=[]
        
        self.A=np.zeros((self.nnx,self.nnx))
        self.b=np.zeros(self.nnx)
        
    def refine(self, r):
        
        self.nex=int(round(r*self.nex))
        self.nnx=(self.nn_el-1)*self.nex+1
        self.nodes=np.linspace(self.domain[0], self.domain[1], num=(self.nnx))
        self.elements=[]
        for i in range(self.nex):
            temp=[]
            for j in range(self.nn_el):
                temp.append(i*(self.nn_el-1)+j)              
            self.elements.append(tuple(temp))
        
        self.reset()
        
    
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
  


# # test2=FEM_2D(((0,10),(0,5)), (11,6))

# # test2.plotmesh()

# # test2.dir_node(11, 0)
# # test2.dir_node(12, 0)

# # test2.plotmesh()

# test=FEM_1D((0,10), 11, interp='quadratic')

# #test.plotmesh()

# c=0.
# a0=-1.
# a1=0.
# a2=1.

# parameters = [c, a0, a1, a2]  # u0 , u , du/dx , d2u/dx2

# # test.dir_node(0, 1)
# # test.dir_node(test.nnx-1, 0)

# test.dir_cond((1,0))


# test.eq_param(parameters)
# #test.axb()

# u = test.solve()

# test.plotmesh()
# plt.show()

# plt.plot(test.nodes, u)
# plt.show()


