# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:52:39 2018

@author: Anatole Gallouet
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

alpha = 0.5

class Node:
    
    
    def __init__(self, x, y, w, children = None):
        """initialize an instance of class Node"""
        self.pos = np.array([x,y])
        self.w = w
        if children is None:
            self.children = []
        else:
            self.children = children
    
    def __str__(self):
        string = "position : (" + str(self.pos[0]) + ", "+ str(self.pos[1]) + ") \n" 
        string += "weight : " + str(self.w)  + "\n"
        return string
    
    def add_child(self, node):
        """add child "node" to the list of children of self"""
        self.children.append(node)
        
    def remove_child(self, node):
        """remove child "node" to the list of children of self"""
        self.children.remove(node)
        
    def reset(self):
        for i in self.children:
            self.remove_child(i)
            i.reset()
    
    def angle(self, node1, node2):
        """compute the angle between vector self-node1 and self,node2"""
        A = self.pos
        B = node1.pos
        C = node2.pos
        return np.arccos(np.dot(B-A, C-A) / (LA.norm(C-A)*LA.norm(B-A))) 
        
    def plot(self):
        self.plot_rec()
        plt.show()
    
    def plot_rec(self):
        """plot the tree self recursively on its children """
        for node in self.children:
            line(self.pos, node.pos, node.w)
            node.plot_rec()
    
    def B_star(self, nodeP, nodeQ):
        """ compute B_star the optimal point for irrigation  from self
        to nodeP and nodeQ by the algorithm described in the article
        nodeP and nodeQ must both be children of self"""     
        O = self.pos
        P = nodeP.pos
        Q = nodeQ.pos
        mO= self.w
        mP = nodeP.w
        mQ = nodeQ.w
        k1 = (mP/mO)**(2*alpha)
        k2 = (mQ/mO)**(2*alpha)
        theta1 = np.arccos((k2 - k1 - 1)/(2*k1**(1/2)))
        theta2 = np.arccos((k1 - k2 - 1)/(2*k2**(1/2)))
        theta3 = np.arccos((1 - k1 - k2)/(2*(k1*k2)**(1/2)))
        
        QM = (np.dot(P-O,Q-O)/np.dot(P-O,P-O)) * (P - O) - (Q - O) 
        PH = (np.dot(P-O,Q-O)/np.dot(Q-O,Q-O)) * (Q - O) - (P - O) 
    
        cot1 = (k2 - k1 - 1) / (4*k1 - (k2 - k1 - 1)**2)**(1/2)
        cot2 = (k1 - k2 - 1) / (4*k2 - (k1 - k2 - 1)**2)**(1/2)
        
        R = (O + P)/2 - (cot1/2)*(LA.norm(P-O) / LA.norm(QM)) * QM;
        S = (O + Q)/2 - (cot2/2)*(LA.norm(Q-O) / LA.norm(PH)) * PH; 
        
        #named lamdba in the paper
        t = np.dot(O-R, S-R)/np.dot(S-R, S-R)
        
        POQ = self.angle(nodeP, nodeQ)
        OQP = nodeQ.angle(self, nodeP)
        OPQ = nodeP.angle(self, nodeQ)
        if (POQ >= theta3):
            return
        elif (OQP >= theta1):
            self.remove_child(nodeP)
            nodeQ.add_child(nodeP)
            nodeQ.w=self.w
            return
        elif (OPQ >= theta2):
            self.remove_child(nodeQ)
            nodeP.add_child(nodeQ)
            nodeP.w=self.w
            return
        else:
            B = 2*((1 - t)*R + t * S) - O
            nodeB = Node(B[0], B[1], self.w, [nodeP, nodeQ])
            self.remove_child(nodeP)
            self.remove_child(nodeQ)
            self.add_child(nodeB)
            return
        
    def B_grad(self, nodeP, nodeQ):
        """ compute B_star the optimal point for irrigation  from self
        to nodeP and nodeQ by the gradient descent method """   
        
        O = self.pos
        P = nodeP.pos
        Q = nodeQ.pos
        mO= self.w
        mP = nodeP.w
        mQ = nodeQ.w

        def DMalpha(x):
            return mO**alpha*(x-O)*1.0/np.linalg.norm(x-O)-mP**alpha*(P-x)*1.0/np.linalg.norm(P-x)-mQ**alpha*(Q-x)*1.0/np.linalg.norm(Q-x)

        step = 0.01
        epsilon = 0.01
        MAX = 10000
        start = (O+P+Q)/3.0
    
        def descent(x, n):
            diff = DMalpha(x)
            if(np.linalg.norm(diff)<epsilon) or n==0:
                return x
            else:
                return descent(x-step*diff, n-1)

        B = descent(start, MAX)
        if (np.linalg.norm(B-O)<epsilon):
            return
        elif (np.linalg.norm(B-Q)<epsilon):
            self.remove_child(nodeP)
            nodeQ.add_child(nodeP)
            nodeQ.w=self.w
            return
        elif (np.linalg.norm(B-P)<epsilon):
            self.remove_child(nodeQ)
            nodeP.add_child(nodeQ)
            nodeP.w=self.w
            return
        else:
            nodeB = Node(B[0], B[1], self.w, [nodeP, nodeQ])
            self.remove_child(nodeP)
            self.remove_child(nodeQ)
            self.add_child(nodeB)
        return
        
    def Malpha(self):
        """ computes Malpha aka the sum to minimize """
        s = 0
        for i in self.children:
            len = np.linalg.norm(self.pos-i.pos)
            s += i.w**alpha * len
            s += i.Malpha()
        return s
        
    
        
            
def line(A, B, weight):
    """draw a line between A and B of witdh proportionnal to weight"""
    plt.plot([A[0],B[0]], [A[1], B[1]], marker = ' ', linewidth = 4*weight, color = '0')
   
"""sorts a set of points in a domain according to subdivisions

points is the set of points to sort in the form of nodes

x is the set of horizontal delimiters and y is the set of vertical delimiters

this function returs: the set of points corresponding to the centers of the new subdomains,
the set of matrices corresponding to the root points in each given domain
and the set of vectors corresponding to the subdomains"""

def sort(points, x, y):
    subpoints = []
    root = []
    subdomain = []
    for i in range(3):
        for j in range(3):
            root.append([(x[i+1]+x[i])/2.0,(y[j+1]+y[j])/2.0])
            subdomain.append([x[i],y[j],x[i+1],y[j+1]])
            sp = []
            for k in points:
                if k.pos[0]>=x[i] and k.pos[0]<x[i+1] and k.pos[1]>=y[j] and k.pos[1]<y[j+1]:
                    sp.append(k)
            subpoints.append(sp)
    return subpoints, root, subdomain
    
    
"""For the subdivision method, we have a list of weighted points and we want to
create in intital transport path

We will thus take the coordienates of the root point

We will also take the list of points, which takes the form of a list of nodes

The first two colums are the coordinates of the point while the third is the weight

The weight of the root point is equal to the sums of the weights of the end points

Finally, we will take the doman vector [xmin, ymin, xmax, ymax]"""

def subdivision(root, points, domain):
    N = len(points)
    if N==1:
        return Node(root[0], root[1], points[0].w, [points[0]])
    if N==2:
        O = Node(root[0], root[1], points[0].w + points[1].w, [points[0],points[1]])
        O.B_star(points[0],points[1])
        return O
    n=2
    l=3
    K=l**n
    if N<=K:
        gmax=0
        i_star=Node(root[0],root[1],0)
        j_star=Node(root[0],root[1],0)
        B_star=Node(root[0],root[1],0)
        for i in points:
            for j in points:
                if(i.pos[0] != j.pos[0] and i.pos[1] != j.pos[1]):
                    O = Node(root[0], root[1], i.w+j.w, [i,j])
                    MO = O.Malpha()
                    O.B_star(i,j)
                    MB = O.Malpha()
                    g = MO - MB
                    if g>=gmax:
                        g=gmax
                        i_star=i
                        j_star=j
                        B_star.reset()
                        if(len(O.children)==1):
                            B_star=O.children[0]
                        else:
                            B_star=O
        points.remove(i_star)
        points.remove(j_star)
        O = subdivision(root, points, domain)
        O.w += B_star.w
        if(O.pos[0] != B_star.pos[0] and O.pos[1] != B_star.pos[1]):
            O.add_child(B_star)
        else:
            O.add_child(i_star)
            O.add_child(j_star)
    xlines=np.linspace(domain[0],domain[2],l+1)
    ylines=np.linspace(domain[1],domain[3],l+1) 
    subpoints, subroot, subdomain = sort(points, xlines, ylines)
    G=[]
    w=0
    for i in range(K):
        if(len(subpoints[i]))>0:
            Gi=subdivision(subroot[i], subpoints[i], subdomain[i])
            w+=Gi.w
            G.append(Gi)
    return Node(root[0],root[1],w,G)

""" Subdivision algorithm applied without the method on a small number of points """

def extsubdivision(root, points, domain):
    N = len(points)
    if N==1:
        return Node(root[0], root[1], points[0].w, [points[0]])
    if N==2:
        O = Node(root[0], root[1], points[0].w + points[1].w, [points[0],points[1]])
        O.B_star(points[0],points[1])
        return O
    n=2
    l=3
    K=l**n
    xlines=np.linspace(domain[0],domain[2],l+1)
    ylines=np.linspace(domain[1],domain[3],l+1) 
    subpoints, subroot, subdomain = sort(points, xlines, ylines)
    G=[]
    w=0
    for i in range(K):
        if(len(subpoints[i]))>0:
            Gi=extsubdivision(subroot[i], subpoints[i], subdomain[i])
            w+=Gi.w
            G.append(Gi)
    return Node(root[0],root[1],w,G)

""" Method on a small number of points applied globally"""

def pairing(root, points):
    N = len(points)
    if N==1:
        return Node(root[0], root[1], points[0].w, [points[0]])
    if N==2:
        O = Node(root[0], root[1], points[0].w + points[1].w, [points[0],points[1]])
        O.B_star(points[0],points[1])
        return O
    gmax=0
    i_star=Node(root[0],root[1],0)
    j_star=Node(root[0],root[1],0)
    B_star=Node(root[0],root[1],0)
    for i in points:
        for j in points:
            if(i.pos[0] != j.pos[0] and i.pos[1] != j.pos[1]):
                O = Node(root[0], root[1], i.w+j.w, [i,j])
                MO = O.Malpha()
                O.B_star(i,j)
                MB = O.Malpha()
                g = MO - MB
                if g>=gmax:
                    g=gmax
                    i_star=i
                    j_star=j
                    B_star.reset()
                    if(len(O.children)==1):
                        B_star=O.children[0]
                    else:
                        B_star=O
    points.remove(i_star)
    points.remove(j_star)
    O = pairing(root, points)
    O.w += B_star.w
    if(O.pos[0] != B_star.pos[0] and O.pos[1] != B_star.pos[1]):
        O.add_child(B_star)
    else:
        O.add_child(i_star)
        O.add_child(j_star)
    return O

"""Chain method, where each node has a child which is the least costly node to it"""

def chain(root, points):
    w = 0
    for i in points:
        w+=i.w
    imin = points[0]
    Omin = Node(root[0], root[1], w, [imin])
    Mmin = Omin.Malpha()
    for i in points:
        O = Node(root[0], root[1], w, [i])
        if (O.Malpha() < Mmin):
            Mmin = O.Malpha()
            imin = i
            Omin = O
    points.remove(imin)
    if (len(points)>0):
        Omin.add_child(chain(imin.pos, points))
    return Omin

""" Naive method"""

def naive(root, points):
    w = 0
    for i in points:
        w+=i.w
    return Node(root[0], root[1], w, points)

"""Average method: averages the weights of all the points and uses that as the
node that branches from the center to every point"""
def average(root, points):
    p = [0,0]
    w = 0
    for i in points:
        w+=i.w
        p += i.pos*i.w**alpha
    p[0] += root[0]*w**alpha
    p[1] += root[1]*w**alpha
    p /= (len(points)+1)
    O = Node(p[0], p[1], w, points)
    return Node(root[0], root[1], w, [O])

"""Average method: averages the weights of all the points and uses that as the
node that branches from the center to every point"""
def average(root, points):
    p = [0,0]
    w = 0
    for i in points:
        w+=i.w
        p += i.pos*i.w**alpha
    p[0] += root[0]*w**alpha
    p[1] += root[1]*w**alpha
    p[0] /= (len(points)+1)
    p[1] /= (len(points)+1)
    O = Node(p[0], p[1], w, points)
    return Node(root[0], root[1], w, [O])

"""Average method applied to the subdivision method """

def averagesubdiv(root, points, domain):
    N=len(points)
    n=2
    l=3
    K=l**n
    if N<=K:
        return average(root, points)
    xlines=np.linspace(domain[0],domain[2],l+1)
    ylines=np.linspace(domain[1],domain[3],l+1) 
    subpoints, subroot, subdomain = sort(points, xlines, ylines)
    G=[]
    w=0
    for i in range(K):
        if(len(subpoints[i]))>0:
            Gi=averagesubdiv(root, subpoints[i], subdomain[i])
            w+=Gi.w
            G.append(Gi)
    return Node(root[0],root[1],w,G)