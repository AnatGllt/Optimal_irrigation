#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 26 20:54:38 2018

@author: admin
"""

import random as rd
from tree import Node
import tree
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
xmin = -10
ymin = -10
xmax = 10
ymax = 10

root = [0,0]

domain = [xmin, ymin, xmax, ymax]

N = 1000

points = []

for i in range(N):
    points.append(Node((xmax-xmin)*rd.random()+xmin, (ymax-ymin)*rd.random()+ymin, rd.random()*8/N))

O = tree.subdivision(root, points, domain)

O.plot()



