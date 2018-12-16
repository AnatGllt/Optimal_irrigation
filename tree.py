
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

alpha = 0.75

class Node:
    
    def __init__(self, x, y, w, father = None, children = None):
        """initialize an instance of class Node"""
        self.pos = np.array([x,y])
        self.w = w
        self.father = father
        f = father
        while f is not None:
            f.w += w
            f = f.father
        if children is None:
            self.children = []
        else:
            self.children = children
            for child in children:
                child.father = self
            if w != sum([P.w for P in children]):
                raise ValueError("Created node has weight different than the sum of its children's")
    
    def check_tree(self):
        res = True
        if self.children != []:
            res = self.w == sum([c.w for c in self.children])
        if not res:
            print(self.w, " != ", sum([c.w for c in self.children]))
        for child in self.children:
            res = res and child.check_tree()
        return res
        
    def  print_tree(self):
        print(self)
        for c in self.children:
            c.print_tree()
    
    def copy(self):
        return Node(self.pos[0], self.pos[1], self.w, None, None)
    
    def __str__(self):
        string = "position : (" + str(self.pos[0]) + ", "+ str(self.pos[1]) + ") \n" 
        string += "weight : " + str(self.w)  + "\n"
        return string
    
    def add_child(self, node):
        """add child "node" to the list of children of self"""
        self.children.append(node)
        node.father = self
        self.w += node.w
        f = self.father
        while f is not None:
            f.w += node.w
            f = f.father
        
    def remove_child(self, node):
        """remove child "node" to the list of children of self"""
        try:
            self.children.remove(node)
        except:
            raise ValueError("Trying to remove a child that isn't one")
        self.w -= node.w
        f = self.father
        while f is not None:
            f.w -= node.w
            f = f.father
        
    def reset(self):
        for i in self.children:
            self.remove_child(i)
            i.reset()
    
    
    def angle(self, node1, node2):
        """compute the angle between vector self-node1 and self,node2"""
        A = self.pos
        B = node1.pos
        C = node2.pos
        return np.arccos(truncate(np.dot(B-A, C-A) / (LA.norm(C-A)*LA.norm(B-A))))
        
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
        if equ(self.pos, nodeP.pos) or equ(self.pos, nodeQ.pos) or equ(nodeQ.pos, nodeP.pos):
            raise ValueError("Two points are the same in computationn of B_star")
        O = self.pos
        P = nodeP.pos
        Q = nodeQ.pos
        if is_aligned(O,P,Q):
            C = middle(O,P,Q)
            if np.all(C == O):
                return False
            if np.all(C == Q):
                self.remove_child(nodeP)
                nodeQ.add_child(nodeP)
                return True
            self.remove_child(nodeQ)
            nodeP.add_child(nodeQ)
            return True
        mO= self.w
        mP = nodeP.w
        mQ = nodeQ.w
        k1 = (mP/mO)**(2*alpha)
        k2 = (mQ/mO)**(2*alpha)
        theta1 = np.arccos(truncate((k2 - k1 - 1)/(2*k1**(1/2))))
        theta2 = np.arccos(truncate((k1 - k2 - 1)/(2*k2**(1/2))))
        theta3 = np.arccos(truncate((1 - k1 - k2)/(2*(k1*k2)**(1/2))))
        
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
            return False
        elif (OQP >= theta1):
            self.remove_child(nodeP)
            nodeQ.add_child(nodeP)
            return True
        elif (OPQ >= theta2):
            self.remove_child(nodeQ)
            nodeP.add_child(nodeQ)
            return True
        else:
            B = 2*((1 - t)*R + t * S) - O
            nodeB = Node(B[0], B[1], 0, self)
            nodeB.add_child(nodeP)
            nodeB.add_child(nodeQ)
            self.remove_child(nodeP)
            self.remove_child(nodeQ)
            self.add_child(nodeB)
            return True

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
            return
        elif (np.linalg.norm(B-P)<epsilon):
            self.remove_child(nodeQ)
            nodeP.add_child(nodeQ)
            return
        else:
            nodeB = Node(B[0], B[1], self.w, self, [nodeP, nodeQ])
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
    
    def local_optimization(self):
        """local optimiaztion of the graph"""
        if len(self.children) == 0:
            return
        points = []
        print(self.children)
        for i in self.children:
            points.append(i)
        for i in self.children:
            self.remove_child(i)
        print(points)
        SNOP(self, points)
        for child in self.children:
            child.local_optimization()

    def Pg(self, t):
        if self.father is None:
            return 0
        return LA.norm(self.pos - self.father.pos)*(self.w**alpha - (self.w - t)**alpha) +  self.father.Pg(t)
    
        
    def potential_father(self, root, sigma):
        """ return a list of potential father  of node self"""
        if self.father is None:
            #global root has no potential father
            return []
        l = []
        if LA.norm(root.pos - self.pos) < sigma:
            l.append(root)
        for node in root.children:
            if node != self.father:
                l += self.potential_father(node, sigma)
        return l
                
    def update_father(self):
        if self.father is None:
            return
        sigma = self.Pg(self.w)/self.w**alpha
        max_gain = sigma * self.w**alpha
        new_father = self.father
        self.father.remove_child(self)
        root = self.father
        while root.father is not None:
            root = root.father
        for node in self.potential_father(root, sigma):
            c = - node.Pg(-self.w)
            if c > max_gain:
                max_gain = c
                new_father = node
        new_father.add_child(self)
        return
    
    def update_all(self):
        self.update_father()
        for child in self.children:
            child.update_father()
        
        
        
#End of class Node

def is_aligned(O, P, Q):
    """ return true if O,P and Q are aligned"""
    return abs(np.dot(O - P,O - Q)) == LA.norm(O - P)*LA.norm(O - Q)

def middle(O, P, Q):
    """return the middle poinr of O,P and Q (only has a meaning if they're aligned)"""
    if np.dot(O - P,O - Q) <= 0:
        return O
    if np.dot(Q - P,Q - O) <= 0:
        return Q
    return P

def equ(a, b):
        return a[0] == b[0] and a[1] == b[1]

def truncate(x):
    if x < -1:
        print("truncation of : ",x)
        return -1
    if x > 1:
        print("truncation of : ",x)
        return 1
    return x

def line(A, B, weight):
    """draw a line between A and B of witdh proportionnal to weight"""
    plt.plot([A[0],B[0]], [A[1], B[1]], marker = ' ', linewidth = 15*weight, color = '0')
    
def sort(points, x, y):
    subpoints = []
    root = []
    subdomain = []
    for i in range(3):
        for j in range(3):
            root.append(Node((x[i+1]+x[i])/2.0, (y[j+1]+y[j])/2.0, 0))
            subdomain.append([x[i],y[j],x[i+1],y[j+1]])
            sp = []
            for k in points:
                if k.pos[0]>=x[i] and k.pos[0]<x[i+1] and k.pos[1]>=y[j] and k.pos[1]<y[j+1]:
                    sp.append(k)
            subpoints.append(sp)
    return subpoints, root, subdomain


def subdivision(root, points, domain):
    """creates the initial graph using the subdivision method"""
    N = len(points)
    l=3
    n=2
    K=l**n
    if N<K:
        return SNOP(root, points)
    xlines=np.linspace(domain[0],domain[2],l+1)
    ylines=np.linspace(domain[1],domain[3],l+1) 
    subpoints, subroot, subdomain = sort(points, xlines, ylines)
    for i in range(K):
        if(len(subpoints[i]))>0:
            Gi=subdivision(subroot[i], subpoints[i], subdomain[i])
            root.add_child(Gi)
    return root



def SNOP(root, points):
    """Method for a small number of points"""
    N = len(points)
    if N==0:
        return
    if N==1:
        root.add_child(points[0])
        return root
    if N==2:
        root.add_child(points[0])
        root.add_child(points[1])
        root.B_star(points[0],points[1])
        return root
    gmax=0
    ind = [0,1]
    #flag = False will mean that B_star hasn't changed anything
    for ind1 in range(N):
        for ind2 in range(ind1+1,N):
            i = points[ind1].copy()
            j = points[ind2].copy()
            if(i.pos[0] != j.pos[0] or i.pos[1] != j.pos[1]):
                O = Node(root.pos[0], root.pos[1], i.w+j.w, None, [i,j])
                MO = O.Malpha()
                O.B_star(i,j)
                MB = O.Malpha()
                g = MO - MB
                print("g = ",g)
                if g>=gmax:
                    gmax=g
                    ind = [ind1,ind2]
            else:
                print("Two points are the same in SNOP method")
    i_star = points[ind[0]]
    j_star = points[ind[1]]
    O = Node(root.pos[0], root.pos[1], i_star.w+j_star.w, None, [i_star,j_star])
    flag = O.B_star(i_star, j_star)
    points.remove(i_star)
    points.remove(j_star)
    B_star = O.children[0]
    if flag:
        points.append(B_star)
    else:
        root.add_child(i_star)
        root.add_child(j_star)
    return SNOP(root, points)

"""Chain method, where each node has a child which is the least costly node to it"""

def chain(root, points):
    if (len(points)==0):
        return root
    imin = points[0]
    root.add_child(imin)
    Mmin = root.Malpha()
    for i in points:
        root.reset()
        root.add_child(i)
        if (root.Malpha() < Mmin):
            Mmin = root.Malpha()
            imin = i
    root.reset()
    root.add_child(imin)
    points.remove(imin)
    imin = chain(imin, points)
    return root

""" Naive method"""

def naive(root, points):
    w = 0
    O = root
    for i in points:
        w+=i.w
        O.add_child(i)
    O.w = w
    return O

"""Average method: averages the weights of all the points and uses that as the
node that branches from the center to every point"""
def average(root, points):
    p = [0,0]
    w = 0
    for i in points:
        w+=i.w
        p += i.pos*i.w**alpha
    p += root.pos*w**alpha
    p /= (len(points)+1)
    O = Node(p[0], p[1], w)
    for i in points:
        O.add_child(i)
    root.add_child(O)
    root.w=w
    return root

"""Average method applied to the subdivision method """

def averagesubdiv(root, points, domain):
    def iteration(points, domain):
        N=len(points)
        n=2
        l=3
        K=l**n
        if N<=K:
            p = [0,0]
            w = 0
            for i in points:
                w+=i.w
                p += i.pos*i.w**alpha
            p += root.pos*w**alpha
            p /= (len(points)+1)
            O = Node(p[0], p[1], w)
            for i in points:
                O.add_child(i)
            root.add_child(O)
            root.w +=w
            return
        xlines=np.linspace(domain[0],domain[2],l+1)
        ylines=np.linspace(domain[1],domain[3],l+1) 
        subpoints, subroot, subdomain = sort(points, xlines, ylines)
        for i in range(K):
            if(len(subpoints[i]))>0:
                iteration(subpoints[i], subdomain[i])
        return
    iteration(points, domain)
    return root
