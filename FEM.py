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
    
    def __init__(self, domain, nex, interp='linear', discr='linear'):
        
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
        
        
        plt.title(f'Discretization for {len(self.elements)} elements, {self.interp} basis functions')
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
                                   - a2*w[p]*x1*phx[m]*phx[n]
                                   + a1*w[p]*x1*ph[m]*phx[n]
                                   + a0*w[p]*x1*ph[m]*ph[n])
                    
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
        self.left_boundary_neu=[]
        self.right_boundary_neu=[]
        self.upper_boundary_neu=[]
        self.lower_boundary_neu=[]
        self.left_boundary_rob=[]
        self.right_boundary_rob=[]
        self.upper_boundary_rob=[]
        self.lower_boundary_rob=[]
        
    
    def eq_param(self, parameters):
        
        self.par=parameters          # constant , u , du/dx , d2u/dx2, du/dy, d2u/dy2
    
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
                    
        for i in range(len(self.elements)):
            
            el=self.elements[i]
            
            add=False
            for n in range(self.nn_el):
                
                temp=[]
                for row in self.neubc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.left_boundary_neu):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                self.left_boundary_neu.append([i, value])
            
            add=False
            for n in range(0, self.nn_el**2-self.nn_el+1, self.nn_el):  
                
                temp=[]
                for row in self.neubc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.lower_boundary_neu):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.lower_boundary_neu.append([i, value])
            
            add=False
            for n in range(self.nn_el-1, self.nn_el**2, self.nn_el):      
                
                temp=[]
                for row in self.neubc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.upper_boundary_neu):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.upper_boundary_neu.append([i, value])
            
            add=False
            for n in range(self.nn_el**2-self.nn_el, self.nn_el**2):     
                
                temp=[]
                for row in self.neubc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.right_boundary_neu):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.right_boundary_neu.append([i, value])
            
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
                    
        for i in range(len(self.elements)):
            
            el=self.elements[i]
            
            add=False
            for n in range(self.nn_el):
                
                temp=[]
                for row in self.robc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.left_boundary_rob):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.left_boundary_rob.append([i, value])
            
            add=False
            for n in range(0, self.nn_el**2-self.nn_el+1, self.nn_el):  
                
                temp=[]
                for row in self.robc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.lower_boundary_rob):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.lower_boundary_rob.append([i, value])
            
            add=False
            for n in range(self.nn_el-1, self.nn_el**2, self.nn_el):      
                
                temp=[]
                for row in self.robc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.upper_boundary_rob):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.upper_boundary_rob.append([i, value])
            
            add=False                    
            for n in range(self.nn_el**2-self.nn_el, self.nn_el**2):     
                
                temp=[]
                for row in self.robc:
                    temp.append(row[0])
                if (el[n] in temp) and (i not in self.right_boundary_rob):
                    
                    add=True
                    
                else:
                    
                    add=False
                    break
                    
            if add==True:
                    self.right_boundary_rob.append([i, value])     
    
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
    
    def dom_transf(self, in_pos, boundaries, corners):
        
        inx, iny = in_pos
        xofy_bound = boundaries[0]
        yofx_bound = boundaries[1]
        ld, rd, ru, lu = corners
        
        scale_x = [rd[0]-ld[0] , ru[0]-lu[0]]
        scale_y = [lu[1]-ld[1] , ru[1]-rd[1]]
        
        x = xofy_bound[0](scale_y[0]*iny) * (1.-inx) + xofy_bound[1](scale_y[1]*iny) * inx
        y = yofx_bound[0](scale_x[0]*inx) * (1.-iny) + yofx_bound[1](scale_x[1]*inx) * iny
        
        return x , y
    
    def to_trapezoid(self, ub, db=None):
        
        if db==None:
            db=self.xdomain[1]*np.array([0.0, 1.])
        
        for i in range(len(self.nodes)):
            
            self.nodes[i]=self.trap_trfm(self.nodes[i], ub, db)
            
    def to_transf(self, boundaries, corners):
        
        for i in range(len(self.nodes)):
            
            self.nodes[i]=self.dom_transf(self.nodes[i], boundaries, corners)
            
    def move(self, vector):
        
        for i in range(len(self.nodes)):
            
            self.nodes[i]+=np.array(vector)
    
    def ph(self, x, y):
        
        if self.interp=='linear':
            
            phx = np.array([1.-x , x])
            phy = np.array([1.-y , y])
            
            return np.array([phx[0]*phy[0],
                             phx[0]*phy[1],
                             phx[1]*phy[0],
                             phx[0]*phy[1]])
        
        elif self.interp=='quadratic':
            
            phx = np.array([2.*x**2 -3.*x+1. ,
                             -4.*x**2 +4.*x ,
                             2.*x**2 -x])
            
            phy = np.array([2.*y**2 -3.*y+1. ,
                             -4.*y**2 +4.*y ,
                             2.*y**2 -y])
            
            return np.array([phx[0]*phy[0],
                             phx[0]*phy[1],
                             phx[0]*phy[2],
                             phx[1]*phy[0],
                             phx[1]*phy[1],
                             phx[1]*phy[2],
                             phx[2]*phy[0],
                             phx[2]*phy[1],
                             phx[2]*phy[2]])
            
    def dph_dx(self, x, y):
        
        if self.interp=='linear':
            
            dphx_dx = np.array([-1. , 1.])
            
            phy = np.array([1.-y , y])
            
            return np.array([dphx_dx[0]*phy[0],
                             dphx_dx[0]*phy[1],
                             dphx_dx[1]*phy[0],
                             dphx_dx[0]*phy[1]])
        
        elif self.interp=='quadratic':
            
            dphx_dx = np.array([4.*x -3. ,
                             -8.*x +4. ,
                             4.*x -1.])
            
            phy = np.array([2.*y**2 -3.*y+1. ,
                             -4.*y**2 +4.*y ,
                             2.*y**2 -y])
            
            return np.array([dphx_dx[0]*phy[0],
                             dphx_dx[0]*phy[1],
                             dphx_dx[0]*phy[2],
                             dphx_dx[1]*phy[0],
                             dphx_dx[1]*phy[1],
                             dphx_dx[1]*phy[2],
                             dphx_dx[2]*phy[0],
                             dphx_dx[2]*phy[1],
                             dphx_dx[2]*phy[2]])
        
    def dph_dy(self, x, y):
        
        if self.interp=='linear':
            
            phx = np.array([1.-x , x])
            
            dphy_dy = np.array([-1. , 1.])
            
            return np.array([phx[0]*dphy_dy[0],
                             phx[0]*dphy_dy[1],
                             phx[1]*dphy_dy[0],
                             phx[0]*dphy_dy[1]])
        
        elif self.interp=='quadratic':
            
            phx = np.array([2.*x**2 -3.*x+1. ,
                             -4.*x**2 +4.*x ,
                             2.*x**2 -x])
            
            dphy_dy = np.array([4.*y -3. ,
                             -8.*y +4. ,
                             4.*y -1.])
            
            return np.array([phx[0]*dphy_dy[0],
                             phx[0]*dphy_dy[1],
                             phx[0]*dphy_dy[2],
                             phx[1]*dphy_dy[0],
                             phx[1]*dphy_dy[1],
                             phx[1]*dphy_dy[2],
                             phx[2]*dphy_dy[0],
                             phx[2]*dphy_dy[1],
                             phx[2]*dphy_dy[2]])        
    
    def axb(self, ngp=None):
        
        self.A=np.zeros((len(self.nodes),len(self.nodes)))
        self.b=np.zeros(len(self.nodes))
        
        for nel in range(len(self.elements)):
            
            self.abfind(nel, ngp)
        
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
        a=self.par[1]
        a1x=self.par[2]
        a2x=self.par[3]
        a1y=self.par[4]
        a2y=self.par[5]
         
        # Neumann BC
        
                #   left Neumann
                
        for element, value in self.left_boundary_neu:
            
            for p in range(self.nn_el):
                
                ksi, ita = 0. , gp[p]
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                y=0.
                x2=0.
                y2=0.
                
                
                for n in range(self.nn_el*self.nn_el):
                    
                    y  = y  + self.nodes[self.nop(n, element), 1] * ph[n]           # y
                    x2 = x2 + self.nodes[self.nop(n, nel), 0] * dph_dita[n]         # dx/dη
                    y2 = y2 + self.nodes[self.nop(n, element), 1] * dph_dita[n]     # dy/dη
                
                if value[1]=='n_in':
                    
                    tot=value[0]
                    value[0] = tot * y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    value[1] = tot * -x2/np.sqrt(x2**2+y2**2)   # value in y direction
                    
                if value[1]=='n_out':
                    
                    tot=value[0]
                    value[0] = tot * -y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    value[1] = tot * x2/np.sqrt(x2**2+y2**2)     # value in y direction
                
                for ln in range(self.nn_el):
                    
                    self.b[self.nop(ln, element)] += + w[p] * a2x * y2 * ph[ln] * value[0]
                    
                    self.b[self.nop(ln, element)] += - w[p] * a2y * x2 * ph[ln] * value[1]
        
                #   lower Neumann
                            
        for element, value in self.lower_boundary_neu:
            
            for p in range(self.nn_el):
                
                ksi, ita = gp[p] , 0.
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                x=0.
                x1=0.
                y1=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    x  = x  + self.nodes[self.nop(n, element), 0] * ph[n]           # x
                    x1 = x1 + self.nodes[self.nop(n, element), 0] * dph_dksi[n]     # dx/dξ
                    y1 = y1 + self.nodes[self.nop(n, nel), 1]     * dph_dksi[n]     # dy/dξ
                
                if value[1]=='n_in':
                    
                    tot=value[0]
                    value[0] = tot * -y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    value[1] = tot * x1/np.sqrt(x1**2+y1**2)     # value in y direction
                    
                if value[1]=='n_out':
                    
                    tot=value[0]
                    value[0] = tot * y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    value[1] = tot * -x1/np.sqrt(x1**2+y1**2)   # value in y direction
                
                for ln in range(0, self.nn_el*self.nn_el-self.nn_el, self.nn_el):
                    
                    self.b[self.nop(ln, element)] += - w[p] * a2y * x1 * ph[ln] * value[1]
                    
                    self.b[self.nop(ln, element)] += + w[p] * a2x * y1 * ph[ln] * value[0]
        
                #   right Neumann
                
        for element, value in self.right_boundary_neu:
            
            for p in range(self.nn_el):
                
                ksi, ita = 1. , gp[p]
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                y=0.
                x2=0.
                y2=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    y  = y  + self.nodes[self.nop(n, element), 1] * ph[n]           # y
                    x2 = x2 + self.nodes[self.nop(n, nel), 0] * dph_dita[n]         # dx/dη
                    y2 = y2 + self.nodes[self.nop(n, element), 1] * dph_dita[n]     # dy/dη
                
                if value[1]=='n_in':
                    
                    tot=value[0]
                    value[0] = tot * -y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    value[1] = tot * x2/np.sqrt(x2**2+y2**2)     # value in y direction
                    
                if value[1]=='n_out':
                    
                    tot=value[0]
                    value[0] = tot * y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    value[1] = tot * -x2/np.sqrt(x2**2+y2**2)   # value in y direction
                
                for ln in range(self.nn_el*self.nn_el-self.nn_el, self.nn_el*self.nn_el):

                    self.b[self.nop(ln, element)] +=  - w[p] * a2x * y2 * ph[ln] * value[0]
                    
                    self.b[self.nop(ln, element)] +=  + w[p] * a2y * x2 * ph[ln] * value[1]
        
                #   upper Neumann
                            
        for element, value in self.upper_boundary_neu:
            
            for p in range(self.nn_el):
                
                ksi, ita = gp[p] , 1.
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                x=0.
                x1=0.
                y1=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    x  = x  + self.nodes[self.nop(n, element), 0] * ph[n]           # x
                    x1 = x1 + self.nodes[self.nop(n, element), 0] * dph_dksi[n]     # dx/dξ
                    y1 = y1 + self.nodes[self.nop(n, nel), 1]     * dph_dksi[n]     # dy/dξ
                
                if value[1]=='n_in':
                    
                    tot=value[0]
                    value[0] = tot * y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    value[1] = tot * -x1/np.sqrt(x1**2+y1**2)   # value in y direction
                    
                if value[1]=='n_out':
                    
                    tot=value[0]
                    value[0] = tot * -y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    value[1] = tot * x1/np.sqrt(x1**2+y1**2)     # value in y direction
                
                for ln in range(self.nn_el-1, self.nn_el*self.nn_el, self.nn_el):
                    
                    self.b[self.nop(ln, element)] += + w[p] * a2y * x1 * ph[ln] * value[1]
                    
                    self.b[self.nop(ln, element)] += - w[p] * a2x * y1 * ph[ln] * value[0]
        
        # Robin BC                                        # continue from here

                #   left Robin
                
        for element, value in self.left_boundary_rob:
            
            for p in range(self.nn_el):
                
                ksi, ita = 0. , gp[p]
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                y=0.
                x2=0.
                y2=0.
                
                
                for n in range(self.nn_el*self.nn_el):
                    
                    y  = y  + self.nodes[self.nop(n, element), 1] * ph[n]           # y
                    x2 = x2 + self.nodes[self.nop(n, nel), 0] * dph_dita[n]         # dx/dη
                    y2 = y2 + self.nodes[self.nop(n, element), 1] * dph_dita[n]     # dy/dη
                
                if value[2]=='n_in':
                    
                    rob1=value[0]     # Robin condition would be   du/dx = rob1 u + rob0
                    rob0=value[1]
                    xdir = y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    ydir = -x2/np.sqrt(x2**2+y2**2)   # value in y direction
                    
                if value[2]=='n_out':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = -y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    ydir = x2/np.sqrt(x2**2+y2**2)     # value in y direction
                
                for ln in range(self.nn_el):
                    
                    self.b[self.nop(ln, element)] += + w[p] * a2x * y2 * ph[ln] * rob0*xdir
                    
                    self.b[self.nop(ln, element)] += - w[p] * a2y * x2 * ph[ln] * rob0*ydir
                    
                    for m in range(self.nn_el*self.nn_el):
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= - w[p] * a2x * y2 * ph[ln]*ph[m] * rob1*xdir
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= + w[p] * a2y * x2 * ph[ln]*ph[m] * rob1*ydir                  
        
                #   lower Robin
                            
        for element, value in self.lower_boundary_rob:
            
            for p in range(self.nn_el):
                
                ksi, ita = gp[p] , 0.
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                x=0.
                x1=0.
                y1=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    x  = x  + self.nodes[self.nop(n, element), 0] * ph[n]           # x
                    x1 = x1 + self.nodes[self.nop(n, element), 0] * dph_dksi[n]     # dx/dξ
                    y1 = y1 + self.nodes[self.nop(n, nel), 1]     * dph_dksi[n]     # dy/dξ
                
                if value[2]=='n_in':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = -y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    ydir = x1/np.sqrt(x1**2+y1**2)     # value in y direction
                    
                if value[2]=='n_out':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    ydir = -x1/np.sqrt(x1**2+y1**2)   # value in y direction
                
                for ln in range(0, self.nn_el*self.nn_el-self.nn_el, self.nn_el):
                    
                    self.b[self.nop(ln, element)] += + w[p] * a2y * x1 * ph[ln] * rob0*ydir
                    
                    self.b[self.nop(ln, element)] += - w[p] * a2x * y1 * ph[ln] * rob0*xdir
                    
                    for m in range(self.nn_el*self.nn_el):
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= - w[p] * a2y * x1 * ph[ln]*ph[m] * rob1*ydir
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= + w[p] * a2x * y1 * ph[ln]*ph[m] * rob1*xdir           
        
                #   right Robin
                
        for element, value in self.right_boundary_rob:
            
            for p in range(self.nn_el):
                
                ksi, ita = 1. , gp[p]
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                y=0.
                x2=0.
                y2=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    y  = y  + self.nodes[self.nop(n, element), 1] * ph[n]           # y
                    x2 = x2 + self.nodes[self.nop(n, nel), 0] * dph_dita[n]         # dx/dη
                    y2 = y2 + self.nodes[self.nop(n, element), 1] * dph_dita[n]     # dy/dη
                
                if value[2]=='n_in':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = -y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    ydir = x2/np.sqrt(x2**2+y2**2)     # value in y direction
                    
                if value[2]=='n_out':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = y2/np.sqrt(x2**2+y2**2)    # value in x direction
                    ydir = -x2/np.sqrt(x2**2+y2**2)   # value in y direction
                
                for ln in range(self.nn_el*self.nn_el-self.nn_el, self.nn_el*self.nn_el):        
                    
                    self.b[self.nop(ln, element)] +=  - w[p] * a2x * y2 * ph[ln] * rob0*xdir
                    
                    self.b[self.nop(ln, element)] +=  + w[p] * a2y * x2 * ph[ln] * rob0*ydir
                    
                    for m in range(self.nn_el*self.nn_el):
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= + w[p] * a2x * y2 * ph[ln]*ph[m] * rob1*xdir
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= - w[p] * a2y * x2 * ph[ln]*ph[m] * rob1*ydir
        
                #   upper Robin
                            
        for element, value in self.upper_boundary_rob:
            
            for p in range(self.nn_el):
                
                ksi, ita = gp[p] , 1.
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                x=0.
                x1=0.
                y1=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    x  = x  + self.nodes[self.nop(n, element), 0] * ph[n]           # x
                    x1 = x1 + self.nodes[self.nop(n, element), 0] * dph_dksi[n]     # dx/dξ
                    y1 = y1 + self.nodes[self.nop(n, nel), 1]     * dph_dksi[n]     # dy/dξ
                
                if value[2]=='n_in':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    ydir = -x1/np.sqrt(x1**2+y1**2)   # value in y direction
                    
                if value[2]=='n_out':
                    
                    rob1=value[0]
                    rob0=value[1]
                    xdir = -y1/np.sqrt(x1**2+y1**2)    # value in x direction
                    ydir = x1/np.sqrt(x1**2+y1**2)     # value in y direction
                
                for ln in range(self.nn_el-1, self.nn_el*self.nn_el, self.nn_el):
                    
                    self.b[self.nop(ln, element)] += - w[p] * a2y * x1 * ph[ln] * rob0*ydir
                    
                    self.b[self.nop(ln, element)] += + w[p] * a2x * y1 * ph[ln] * rob0*xdir
                    
                    for m in range(self.nn_el*self.nn_el):
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= + w[p] * a2y * x1 * ph[ln]*ph[m] * rob1*ydir
                        
                        self.A[self.nop(ln, element), self.nop(m, element)]+= - w[p] * a2x * y1 * ph[ln]*ph[m] * rob1*xdir
        
        
        
        # Dirichlet BC
        
        for node, value in self.dirbc:
            
            self.A[node]=0.
            
            self.A[node, node]=1.
            
            self.b[node]=value
            
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
        a=self.par[1]
        a1x=self.par[2]
        a2x=self.par[3]
        a1y=self.par[4]
        a2y=self.par[5]
        
        #   loop over gauss points
        for p in range(len(gp)):
            
            for q in range(len(gp)):
                
                ksi, ita = gp[p], gp[q]
                ph = self.ph(ksi, ita)
                dph_dksi = self.dph_dx(ksi, ita)
                dph_dita = self.dph_dy(ksi, ita)
                
                #   isoparametric transformation & x, y
                
                x=0.
                y=0.
                x1=0.
                x2=0.
                y1=0.
                y2=0.
                
                for n in range(self.nn_el*self.nn_el):
                    
                    x  = x  + self.nodes[self.nop(n, nel), 0] * ph[n]           # x
                    y  = y  + self.nodes[self.nop(n, nel), 1] * ph[n]           # y
                    x1 = x1 + self.nodes[self.nop(n, nel), 0] * dph_dksi[n]     # dx/dξ
                    x2 = x2 + self.nodes[self.nop(n, nel), 0] * dph_dita[n]     # dx/dη
                    y1 = y1 + self.nodes[self.nop(n, nel), 1] * dph_dksi[n]     # dy/dξ
                    y2 = y2 + self.nodes[self.nop(n, nel), 1] * dph_dita[n]     # dy/dη
                    
                
                dett = x1 * y2 - x2 * y1
                
                dph_dx = np.zeros(self.nn_el*self.nn_el)
                dph_dy = np.zeros(self.nn_el*self.nn_el)
                
                for k in range(self.nn_el*self.nn_el):
                    
                    dph_dx[k] = ( y2 * dph_dksi[k] - y1 * dph_dita[k] ) / dett
                    dph_dy[k] = ( x1 * dph_dita[k] - y1 * dph_dksi[k] ) / dett
                
                #   Residuals
                
                for l in range(self.nn_el*self.nn_el):
                    
                    l1 = self.nop(l, nel)
                    
                    for m in range(self.nn_el*self.nn_el):
                        
                        m1 = self.nop(m, nel)
                        
                        self.A[l1,m1]=(self.A[l1,m1]
                                       -a2x      * w[p]*w[q] * dett * dph_dx[l]*dph_dx[m]
                                       -a2y      * w[p]*w[q] * dett * dph_dy[l]*dph_dy[m]
                                       +a1x(x,y) * w[p]*w[q] * dett * ph[l]*dph_dx[m]
                                       +a1y(x,y) * w[p]*w[q] * dett * ph[l]*dph_dy[m]
                                       +a        * w[p]*w[q] * dett * ph[l]*ph[m]
                                       )
                        
                    self.b[l1]=(self.b[l1]
                                -c * w[p]*w[q] * dett * ph[l]
                                ) 
       
    def solve(self, ngp=None):
        
        try:
            
            sol = np.linalg.solve(self.A, self.b)
            
        except:
            
            self.axb(ngp)
            
            sol = np.linalg.solve(self.A, self.b)
            
        self.solution=sol
        
        return sol
    
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
            temp=[]
            for row in self.neubc:
                temp.append(row[0])
            plt.scatter(self.nodes[np.array(temp, dtype=int)][:,0], self.nodes[np.array(temp, dtype=int)][:,1], color='yellow', label='Neumann',zorder=10)
        
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

    def plot_cont(self, xlabel, ylabel, zlabel, title, levels):
        
        from scipy.interpolate import griddata
        from scipy.spatial import Delaunay
        
        x = self.nodes[:,0]
        y = self.nodes[:,1]
        z = self.solution
        
        xi = np.linspace(min(x), max(x), 100)  
        yi = np.linspace(min(y), max(y), 100)  
        X, Y = np.meshgrid(xi, yi)             
        
        Z = griddata((x, y), z, (X, Y), method='cubic')
        
        # Plot the contour
        plt.figure(figsize=(12, 6))
        contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')  # Contour lines
        plt.clabel(contour, inline=True, fontsize=8)               # Label the contours
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Add a color bar
        plt.colorbar(contour, label=zlabel)
        
        # Show the plot
        plt.show()
        
class FEM_1D_nl:
    
    #import numpy as np
    import matplotlib.pyplot as plt
    import autograd.numpy as np
    from autograd import grad
    
    def __init__(self, domain, nex, interp='linear', discr='linear'):
        
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
        
        #self.soltest=np.zeros(self.nnx)
        #self.soltest=np.ones(self.nnx)
        self.soltest=np.random.rand(self.nnx)
        
        self.intest=np.copy(self.soltest)
        
    def eq_param(self, parameters):
        
        self.par=parameters          # multiplier of d2u/dx2 , multiplier of du/dx , f(u)
    
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
        
        
        a2 = self.par[0]
        a1 = self.par[1]
        func  = self.par[2]
        
        utest = self.soltest
        
        #   loop over gauss points
        for p in range(len(gp)):
            
            ph = self.ph(gp[p])
            phd = self.phd(gp[p])
            
            x=0.
            x1=0.
            
            def f(ut):
                u=0.           
                for n in range(self.nn_el):
                    u=u+ut[self.nop(n,nel)]*ph[n]
                
                return func(u)
            
            def df_du(ut):
                from autograd import elementwise_grad as grad
                grd = grad(f)(ut)
                
                #print(grd)
                return grd
            
            
            
            for n in range(self.nn_el):
                
                x=x+self.nodes[self.nop(n,nel)]*ph[n]
                x1=x1+self.nodes[self.nop(n,nel)]*phd[n] # dx / dξ
            
            phx = phd/x1
            
            for m in range(self.nn_el):
                
                m1=self.nop(m, nel)
                
                for n in range(self.nn_el):
                    
                    n1=self.nop(n, nel)
                    
                    self.Jac[m1,n1]+=(
                                       - a2*w[p]*x1*phx[m]*phx[n]
                                       + a1*w[p]*x1*ph[m] *phx[n]
                                       + w[p]*x1*ph[m] * df_du(utest)[n]
                                       )
                    
                    self.Rit[m1]+=(
                                    - a2*w[p]*x1*phx[m]*phx[n]*utest[n1]
                                    + a1*w[p]*x1*ph[m] *phx[n]*utest[n1]
                                    )
                self.Rit[m1]+=  + w[p]*x1*ph[m] * f(utest)
                                
    
    def axb(self, ngp=None):
        
        self.Jac=np.zeros((self.nnx,self.nnx))
        self.Rit=np.zeros(self.nnx)
        
        for nel in range(len(self.elements)):
            
            self.abfind(nel, ngp)
        
        
        a2 = self.par[0]
        a1 = self.par[1]
        f  = self.par[2]
        
        
        # Dirichlet BC
        
        for node, value in self.dirbc:
            
            self.Jac[node]=0.
            
            self.Jac[node, node]=1.
            
            self.Rit[node]=np.copy(self.soltest[node])-value
        
        # Neumann BC
        
        for node, value in self.neubc:
            
            if node==0:
                self.Rit[node]+= - a2 * value
                
            if node==(self.nnx-1):
                
                self.Rit[node]+= + a2 * value
                
        # Robin BC     not done
        
        # for node, values in self.robc:
            
        #     rob1, rob0 = values # Robin condition would be   du/dx = rob1 u + rob0
            
        #     if node==0:
        #         self.A[node,node]=self.A[node,node] - a2 * rob1
        #         self.b[node]=self.b[node] + a2 * rob0
                
        #     if node==(self.nnx-1):
        #         self.A[node,node]=self.A[node,node] + a2 * rob1
        #         self.b[node]=self.b[node] - a2 * rob0
            
    def solve(self, max_it=100, epsilon=10**(-4), ngp=None):
        
        self.intest=np.copy(self.soltest)
        
        conv = False
        iteration=0
        
        while(conv==False and iteration<100):
            
            self.axb(ngp)

            du = np.linalg.solve(self.Jac, -self.Rit)
            
            self.soltest += du
            
            if iteration>0:
                if np.sqrt(np.mean((self.soltest-self.prevtest)**2))<epsilon:
                    conv=True
                    break
            
            self.prevtest=np.copy(self.soltest)
            iteration+=1
            
        if conv==True:   
            self.solution=np.copy(self.soltest)
            print(f'Solved after {iteration} iterations')
            print(f'RMS for Residuals is {np.sqrt(np.mean((self.Rit)**2))}')
            self.last_sol=np.copy(self.soltest)
            self.last_its_needed=iteration
        else:
            self.solution=None
            print(f'Not converging in {iteration} iterations')
            print(f'RMS for Residuals is {np.sqrt(np.mean((self.Rit)**2))}')
    
    
    def solve_par_cont(self, par, max_it=100, epsilon=10**(-4), ngp=None):
        
        self.intest=np.copy(self.soltest)
        import types
        from functools import partial
        
        if isinstance(self.par[2], partial):
            original_function = self.par[2].func  # Extract the underlying function
        else:
            original_function = self.par[2]

        # Create an independent copy of the function
        func_par = types.FunctionType(
            original_function.__code__,  # Use the code object of the underlying function
            original_function.__globals__,  # Use the same global namespace
            name=original_function.__name__,  # Copy the name of the function
            argdefs=original_function.__defaults__,  # Copy the default arguments
            closure=original_function.__closure__  # Copy the closure (if any)
            )
        

        self.par[2]=partial(self.par[2], par=par)
        
        conv = False
        iteration=0
        
        while(conv==False and iteration<100):
            
            self.axb(ngp)

            du = np.linalg.solve(self.Jac, -self.Rit)
            
            self.soltest += du
            
            if iteration>0:
                if np.sqrt(np.mean((self.soltest-self.prevtest)**2))<epsilon:
                    conv=True
                    break
            
            self.prevtest=np.copy(self.soltest)
            iteration+=1
            
        if conv==True:   
            self.solution=np.copy(self.soltest)
            print(f'Solved after {iteration} iterations')
            print(f'RMS for Residuals is {np.sqrt(np.mean((self.Rit)**2))}')
            self.last_sol=np.copy(self.soltest)
            self.last_its_needed=iteration
        else:
            self.solution=None
            print(f'Not converging in {iteration} iterations')
            print(f'RMS for Residuals is {np.sqrt(np.mean((self.Rit)**2))}')
        
        res=np.zeros(self.nnx)
        
        for nel in range(len(self.elements)):
            
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
            
            
            a2 = self.par[0]
            a1 = self.par[1]

            func  = func_par
            
            utest = np.copy(self.soltest)
            
            #   loop over gauss points
            for p in range(len(gp)):
                
                ph = self.ph(gp[p])
                phd = self.phd(gp[p])
                
                x=0.
                x1=0.
                
                def f(par):
                    u=0.           
                    for n in range(self.nn_el):
                        u=u+utest[self.nop(n,nel)]*ph[n]
                    
                    return func(u,par)
                
                def df_du(par):
                    from autograd import grad
                    from functools import partial
                    grd = grad(f)(par)
                    
                    #print(grd)
                    return grd
                
                
                for n in range(self.nn_el):
                    
                    x=x+self.nodes[self.nop(n,nel)]*ph[n]
                    x1=x1+self.nodes[self.nop(n,nel)]*phd[n] # dx / dξ
                
                phx = phd/x1
                
                for m in range(self.nn_el):
                    
                    m1=self.nop(m, nel)
                    
                    res[m1]= res[m1]  + w[p]*x1*ph[m] * df_du(par)
        
        self.dR_dpar= res

        
    
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