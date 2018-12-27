#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:02:28 2018
@author: admin
"""

import random as rd
import tree_sphere
import math
from tree_sphere import Node
from tree_sphere import SphereNode
from tree_sphere import subdivision
import sys

"""
sys.setrecursionlimit(20000)
weight = rd.random()
P = Node(10*rd.random()-5, 10*rd.random()-5, weight)
Q = Node(10*rd.random()-5, 10*rd.random()-5, 1 - weight)
O = Node(0, 0, 1, [P,Q])
O.B_star(P,Q)
O.plot()
print(O.Malpha())
O.reset()
O = Node(0, 0, 1, [P,Q])
O.B_grad(P,Q)
O.plot()
print(O.Malpha())
"""

"""
xmin = -10
ymin = -10
xmax = 10
ymax = 10

domain = [xmin, ymin, xmax, ymax]

N = 100

points = []
for i in range(N):
    points.append(Node((xmax-xmin)*rd.random()+xmin, (ymax-ymin)*rd.random()+ymin, rd.random()/N))

root = Node(0, 0, 0)

O = subdivision(root, points, domain)

#O = tree.chain(root, points)

#O = tree.naive(root, points)

#O = tree.average(root, points)

#O = tree.averagesubdiv(root, points, domain)

O.plot()

plt.clf()
"""
sys.setrecursionlimit(20000)

P = SphereNode(rd.random()*math.pi*2,rd.random()*math.pi,rd.random())
Q = SphereNode(rd.random()*math.pi*2,rd.random()*math.pi,rd.random())
O = SphereNode(0, 0, 0)
O.add_child(P)
O.add_child(Q)
O.B_star(P,Q)
print(O.Malpha())
#O.plot()

O.reset()
O = SphereNode(0, 0, 0)
O.add_child(P)
O.add_child(Q)
O.B_grad(P,Q)
print(O.Malpha())
#O.plot()