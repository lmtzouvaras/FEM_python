# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

class FEM_1D:
    
    def __init__(self, domain, nex, interp='linear'):
        
        self.nn_el=2
        self.domain=domain
        self.nex=nex
        self.nodes=np.linspace(domain[0], domain[1], num=(nex*self.nn_el-1))
        self.elements=[]
        for i in range(self.nex):
            self.elements.append((i,i+1))
        self.dirbc=[]
        self.neubc=[]
        
    
    def nop(self, ln, el):
        
        return self.elements[el][ln]
    
    def dir_node(self, node, value):
        
        self.dirbc.append([node, value])
    
    
    def plotmesh(self):

        plt.scatter(self.nodes, np.zeros(len(self.nodes)), color='blue')
        
        
        if len(self.dirbc)>0:
            plt.scatter(self.nodes[self.dirbc[:][0]][0], np.zeros(len(self.dirbc)), color='red')
        plt.show()
    
class FEM_2D:
    
     
    def __init__(self, domain, ne, interp='linear'):
        
        self.nn_el=4
        self.domain=domain
        self.xdomain=domain[0]
        self.ydomain=domain[1]
        self.nex=ne[0]
        self.ney=ne[1]
        self.nnx=self.nex*2-1
        self.nny=self.ney*2-1     
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
            plt.scatter(test2.nodes[np.array(test2.dirbc)[:,0],0], test2.nodes[np.array(test2.dirbc)[:,0],1], color='red')
        plt.show()
        
test=FEM_1D((0,10), 6)

# test.plotmesh()

test2=FEM_2D(((0,10),(0,5)), (11,6))

test2.plotmesh()


print(test2.elements)

# for element in test2.elements:
#     for node in element:
#         print(test2.nodes[node])
#     print()

print(test2.elements[2])

print(test2.nop(3, 2))

test.dir_node(0, 0)



test.plotmesh()

test2.dir_node(11, 0)
test2.dir_node(12, 0)


print(np.array(test2.dirbc)[:,0])

print(test2.nodes[np.array(test2.dirbc)[:,0]])

print(test2.nodes)
print(test2.nodes[np.array(test2.dirbc)[:,0],0])

test2.plotmesh()