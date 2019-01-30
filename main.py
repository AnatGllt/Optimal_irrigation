#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:02:28 2018
@author: admin
"""

import random as rd
import tree
from tree import Node
from tree import subdivision
from tree import circle
from tree import rand
import sys
import matplotlib.pyplot as plt

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

domain = [xmin, ymin, xmax, ymax]

N = 50


root, points = circle(N)
#root, points = rand(N)


O = subdivision(root, points, domain)
#O = tree.averagesubdiv(root, points, domain)

O.plot()



print(O.Malpha())

#O = tree.chain(root, points)

#O = tree.naive(root, points)

#O = tree.average(root, points)

#O = tree.averagesubdiv(root, points, domain)

#O.local_optimization()


#O.plot()

#print(O.Malpha())

brute = False
#O.update_all_improved()
#O.update_all_brute()
if brute :
	print("Using brute algo")
	O.global_optimization_brute()
else :
	print("Using paper algo")
	previousMalpha = O.Malpha()
	loop = False
	maxIter = 100
	iter = 0
	while not loop and iter < maxIter:
	#for i in range(2):
		O.global_optimization_improved()
		## Stopping conditions
		loop = (previousMalpha == O.Malpha())
		iter +=1
		previousMalpha = O.Malpha()
	if iter == maxIter:
		print("out because too much iterations")
O.plot()

print(O.Malpha())

"""O.subdivide_edges(3)

O.plot()

O.update_all()

O.plot()

print(O.Malpha())"""

plt.clf()


