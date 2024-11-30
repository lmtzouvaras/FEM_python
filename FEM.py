# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

class FEM_1D:
    
    import numpy as np
    import matplotlib.pyplot as plt
    
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
            
        self.solution=sol
        
        return sol
    
    def dsol_dx(self):
        
        try:
            
            return self.sol_deriv
        
        except:
            
            der=np.zeros_like(self.solution)
            
            for nel in range(len(self.elements)):
                
                for n in range(self.nn_el):
                    xloc=self.nodes[self.nop(n, nel)]-self.nodes[self.nop(0, nel)]
                    xloc=xloc/(self.nodes[self.nop((self.nn_el-1), nel)]-self.nodes[self.nop(0, nel)])
                    phd=self.phd(xloc) 
                    x1=0
                    
                    for k in range(self.nn_el):
                        
                        x1=x1+self.nodes[self.nop(k,nel)]*phd[k] # dx / dξ
                    
                    phx = phd/x1
                    
                    for m in range(self.nn_el):
                        
                        der[self.nop(n, nel)] += self.solution[self.nop(m, nel)] * phx[m]
            
            for nel in range(len(self.elements)-1):
                
                der[self.nop(self.nn_el-1, nel)] = der[self.nop(self.nn_el-1, nel)]/2
            
            self.sol_deriv=der
            
            return der
    
    def reset(self):
        
        self.dirbc=[]
        self.neubc=[]
        self.robc=[]
        
        self.A=np.zeros((self.nnx,self.nnx))
        self.b=np.zeros(self.nnx)
        
        del self.solution
        del self.sol_deriv
        
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
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def __init__(self, domain, ne, interp='linear'):
        
        self.interp=interp
        
        if interp=='linear':
            self.nn_el=2
        elif interp=='quadratic':
            self.nn_el=3
        elif interp=='cubic':
            self.nn_el=5

        self.domain=domain
        self.xdomain=domain[0]
        self.ydomain=domain[1]
        
        self.ne=ne[0]*ne[1]
        self.nex=ne[0]
        self.nnx=(self.nn_el-1)*self.nex+1
        self.ney=ne[1]
        self.nny=(self.nn_el-1)*self.ney+1
        
        self.xnodes=np.linspace(self.xdomain[0], self.xdomain[1], num=self.nnx)
        self.ynodes=np.linspace(self.ydomain[0], self.ydomain[1], num=self.nny)
        self.nodes=np.meshgrid(self.xnodes, self.ynodes)
        self.nodes=np.moveaxis(np.array(self.nodes), 0, self.nodes[0].ndim).reshape(-1, len(self.nodes))
        self.nodes=self.nodes[self.nodes[:,1].argsort(kind='mergesort')]
        self.nodes=self.nodes[self.nodes[:,0].argsort(kind='mergesort')]
        self.elements=[]
        
        for i in range(self.nex):
            for j in range(self.ney):
                temp=[]
                for ii in range(self.nn_el):
                    for jj in range(self.nn_el):
                        pass
                        temp.append(i*self.nny*(self.nn_el-1) 
                                    + j*(self.nn_el-1) 
                                    + ii*self.nny
                                    + jj)
                    
                
                self.elements.append(tuple(temp))
                
        self.dirbc=[]
        self.neubc=[]
        self.robc=[]
    
    def eq_param(self, parameters):
        
        self.par=parameters          # u0 , u , du/dx , d2u/dx2, du/dy, d2u/dy2
    
    def nop(self, ln, el):
        
        return self.elements[el][ln]
    
    def dir_node(self, node, value):
        
        self.dirbc.append([node, value])
        
    def neu_node(self, node, value):
        
        self.neubc.append([node, value])
    
    def rob_node(self, node, value):
        
        self.robc.append([node, value])
        
    def dir_cond(self, position=(None,None), value=None):
        
        if (position[0]!=None and position[1]!=None):
            for i in range(len(self.nodes)):
                if (self.nodes[i][0]==position[0] and self.nodes[i][1]==position[1]):
                    self.dir_node(i, value)
                    
        elif (position[0]!=None and position[1]==None):
            for i in range(len(self.nodes)):
                if self.nodes[i][0]==position[0]:
                    self.dir_node(i, value)
                    
        elif (position[0]==None and position[1]!=None):
            for i in range(len(self.nodes)):
                if self.nodes[i][1]==position[1]:
                    self.dir_node(i, value)
            
    def neu_cond(self, position=(None,None), value=None):
        
        if (position[0]!=None and position[1]!=None):
            for i in range(len(self.nodes)):
                if (self.nodes[i][0]==position[0] and self.nodes[i][1]==position[1]):
                    self.neu_node(i, value)
                    
        elif (position[0]!=None and position[1]==None):
            for i in range(len(self.nodes)):
                if self.nodes[i][0]==position[0]:
                    self.neu_node(i, value)
                    
        elif (position[0]==None and position[1]!=None):
            for i in range(len(self.nodes)):
                if self.nodes[i][1]==position[1]:
                    self.neu_node(i, value)
            
    def rob_cond(self, position=(None,None), value=None):
        
        if (position[0]!=None and position[1]!=None):
            for i in range(len(self.nodes)):
                if (self.nodes[i][0]==position[0] and self.nodes[i][1]==position[1]):
                    self.rob_node(i, value)
                    
        elif (position[0]!=None and position[1]==None):
            for i in range(len(self.nodes)):
                if self.nodes[i][0]==position[0]:
                    self.rob_node(i, value)
                    
        elif (position[0]==None and position[1]!=None):
            for i in range(len(self.nodes)):
                if self.nodes[i][1]==position[1]:
                    self.rob_node(i, value)
    
    def trap_trfm(self, vector, ub, db):
        
        height=self.ydomain[-1]
        in_width=self.xdomain[-1]
        
        sc_ub=(ub[1]-ub[0])/in_width
        sc_db=(db[1]-db[0])/in_width
        
        def K(vector):
            
            tr = np.array([[sc_ub*vector[1]/height + sc_db*(1-vector[1]/height), 0.],
                             [0.                                    , 1.]])
            
            return tr
        
        return K(vector).dot(vector) + np.array([ub[0]*vector[1]/height + db[0]*(1-vector[1]/height), 0.])    
    
    def to_trapezoid(self, ub, db=None):
        
        if db==None:
            db=self.xdomain[1]*np.array([0.0, 1.])
        
        for i in range(len(self.nodes)):
            
            self.nodes[i]=self.trap_trfm(self.nodes[i], ub, db)
    
    def plotmesh(self, show_elements=False, grid=False):
    
        plt.scatter(self.nodes[:,0], self.nodes[:,1], color='blue', label='No BC',zorder=5)
        
        if show_elements==True:
            for nel in range(len(self.elements)):
                bnodes = [self.nop(0, nel),
                          self.nop(self.nn_el-1, nel),
                          self.nop(self.nn_el*self.nn_el-1, nel),
                          self.nop(2*self.nn_el, nel),
                          self.nop(0, nel)]
                x=[]
                y=[]
                for n in bnodes:
                    x.append(self.nodes[n][0])
                    y.append(self.nodes[n][1])
                plt.plot(x, y, color='black',zorder=0)
        
        #   Color BC
        
        if len(self.neubc)>0:
            plt.scatter(self.nodes[np.array(self.neubc, dtype=int)[:,0]][:,0], self.nodes[np.array(self.neubc, dtype=int)[:,0]][:,1], color='yellow', label='Neumann',zorder=10)
        
        if len(self.robc)>0:
            for n, val in self.robc:

                plt.scatter(self.nodes[n,0], self.nodes[n,1], color='orange',zorder=10)
            plt.scatter(self.nodes[n,0], self.nodes[n,1], color='orange', label='Robin',zorder=10)
        if len(self.dirbc)>0:
            plt.scatter(self.nodes[np.array(self.dirbc, dtype=int)[:,0]][:,0], self.nodes[np.array(self.dirbc, dtype=int)[:,0]][:,1], color='red', label='Dirichlet',zorder=20)
        
        if grid==True:
            plt.grid()
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()
