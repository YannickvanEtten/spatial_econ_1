from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.optimize
from math import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from itertools import cycle

#Group 2: δ = 0.35; ρ = 0.75; Τ = 1.9. 

phi1=0.48
# phi2=0.6
gam=0.3
L=2.0
eps=5.0
rho=0.75        #changed for gr2
bet=0.8
alp=0.08
delta=0.35      #changed for gr2
T_lambda = []
T_T = []
T_Stab=[]
fig = plt.figure()
lines = ["-","--","-.",":","+","."]
color = ["Black","Green","Red","Yellow","Purple","Grey"]
colstab = ["Red","Black"]
indx = -1
file = open("Results.txt","w")

# Define number of iterations T and lambda
Th_min=150
Th_max=201 
Th_step=1
Th_div=100

lamh_min=1
lamh_max=99
lamh_step=1
lamh_div=100

for Th in range (Th_min,Th_max,Th_step):
    indx += 1
    T=Th/Th_div
    print ("T is equal to", T)
    lamda = []
    Relative = []
    Welfare = []
    W_Man_H = []
    W_Man_F = []
    W_Farm_H = [] 
    W_FRamF = []

    for lamh in range (lamh_min,lamh_max,lamh_step):
        lam  = lamh/lamh_div

        def equations(p):
            Y1, Y2, W1, W2, I1, I2 = p
            return(Y1-phi1*(1-gam)*L-lam*gam*L*W1,
                   Y2-(1-phi1)*(1-gam)*L-(1-lam)*gam*L*W2,
                   W1-rho*bet**(-rho)*(delta/(alp*(eps-1)))**(1/eps)*(Y1*I1**(eps-1)+T**(1-eps)*Y2*I2**(eps-1))**(1/eps),
                   W2-rho*bet**(-rho)*(delta/(alp*(eps-1)))**(1/eps)*(T**(1-eps)*Y1*I1**(eps-1)+Y2*I2**(eps-1))**(1/eps),
                   I1-(gam*L/(alp*eps))**(1/(1-eps))*(bet/rho)*(lam*W1**(1-eps)+(1-lam)*T**(1-eps)*W2**(1-eps))**(1/(1-eps)),
                   I2-(gam*L/(alp*eps))**(1/(1-eps))*(bet/rho)*(lam*T**(1-eps)*W1**(1-eps)+(1-lam)*W2**(1-eps))**(1/(1-eps)))

        Y1, Y2, W1, W2, I1, I2 = fsolve(equations, (1, 1, 1, 1, 1, 1),xtol=1e-10)
        Rel = (W1/I1**delta)/(W2/I2**delta)
        Welf = Y1/(I1**delta)+Y2/(I2**delta)
        Man_H=W1/I1**delta
        Man_F=W2/I2**delta
        Farm_H=1/I1**delta
        Farm_F=1/I2**delta

        lamda.append(lam)
        Welfare.append(Welf)
        Relative.append(Rel)
        W_Man_H.append(Man_H)
        W_Man_F.append(Man_F)
        W_Farm_H.append(Farm_H)
        W_FRamF.append(Farm_F)

    ax = fig.add_subplot(2,3,1)
    if indx == 0:
        ax.plot(lamda,Relative,lines[indx],marker="o",color=color[indx],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(Th_step)):
        ax.plot(lamda,Relative,lines[4],marker="o",color=color[4],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(2*Th_step)-0.5,0):
        ax.plot(lamda,Relative,lines[2],marker="o",color=color[2],markersize=1,label=T)
    else:
        ax.plot(lamda,Relative,lines[5],marker="o",color=color[5],markersize=0.1)
    plt.plot([0, 1], [1, 1], 'k-',lw=0.5,color="Black")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Relative real wage')
    plt.xlabel('Lambda')
    plt.title('Wiggle diagram')

    ax2 = fig.add_subplot(2,3,3)
    if indx ==0:
        ax2.plot(lamda,Welfare,lines[indx],marker="o",color=color[indx],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(Th_step)):
        ax2.plot(lamda,Welfare,lines[4],marker="o",color=color[4],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(2*Th_step),0):
        ax2.plot(lamda,Welfare,lines[2],marker="o",color=color[2],markersize=1,label=T)
    else:
        ax2.plot(lamda,Welfare,lines[5],marker="o",color=color[5],markersize=0.01)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Welfare')
    plt.xlabel('Lambda')
    plt.title('Welfare')

    ax3 = fig.add_subplot(2,3,2)
    if indx ==0:
        ax3.plot(lamda,W_Man_H,lines[indx],marker="o",color=color[indx],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(Th_step)):
        ax3.plot(lamda,W_Man_H,lines[4],marker="o",color=color[4],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(2*Th_step),0):
        ax3.plot(lamda,W_Man_H,lines[2],marker="o",color=color[2],markersize=1,label=T)
    else:
        ax3.plot(lamda,W_Man_H,lines[5],marker="o",color=color[5],markersize=0.1)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('W_Man_H')
    plt.xlabel('Lambda')
    plt.title('Wage manufacturing workers home')
    
    ax4 = fig.add_subplot(2,3,5)
    if indx ==0:
        ax4.plot(lamda,W_Man_F,lines[indx],marker="o",color=color[indx],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(Th_step)):
        ax4.plot(lamda,W_Man_F,lines[4],marker="o",color=color[4],markersize=1,label=T)
    elif indx == round((Th_max-Th_min-1)/(2*Th_step),0):
        ax4.plot(lamda,W_Man_F,lines[2],marker="o",color=color[2],markersize=1,label=T)
    else:
        ax4.plot(lamda,W_Man_F,lines[5],marker="o",color=color[5],markersize=0.1)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
    plt.ylabel('W_Man_F')
    plt.xlabel('Lambda')
    plt.title('Wage manufacturing workers Foreign')

    prec = 0.0005

    for x in range(len(lamda)):
        y = x-1
        if Relative[x] >= 1-prec and Relative[x] <= 1+prec and Relative[x]>Relative[y]:
            axis = x/lamh_div +lamh_min/lamh_div
            #print (axis)
            file.write(str(T) + ": " + str(indx) + ": " + str(axis) + "\n")
            T_lambda.append(axis)
            T_T.append(T)
            T_Stab.append(0) # stability still to be defined by checking whether Relative(x-1)<Relative(x) if so then unstable; same can be done for Relative (x+1)
        elif Relative[x] >= 1-prec and Relative[x] <= 1+prec and Relative[x]<=Relative[y]:
            axis = x/lamh_div +lamh_min/lamh_div
            #print (axis)
            file.write(str(T) + ": " + str(indx) + ": " + str(axis) + "\n")
            T_lambda.append(axis)
            T_T.append(T)
            T_Stab.append(1) # stability still to be defined by checking whether Relative(x-1)<Relative(x) if so then unstable; same can be done for Relative (x+1)
            
        if Relative[lamh_min] < 1:
            axis = 0.0
            #print (axis)
            file.write(str(T) + ": " + str(indx) + ": " + str(axis) + "\n")
            T_lambda.append(axis)
            T_T.append(T)
            T_Stab.append(1)
        else:
            T_lambda.append(0)
            T_T.append(T)
            T_Stab.append(0)
        if Relative[lamh_max-lamh_min-1] > 1:
            axis = 1.0
            #print (axis)
            file.write(str(T) + ": " + str(indx) + ": " + str(axis) + "\n")
            T_lambda.append(axis)
            T_T.append(T)
            T_Stab.append(1)
        else:
            T_lambda.append(1)
            T_T.append(T)
            T_Stab.append(0)
                       
# print (T_lambda,T_T,T_Stab)

file.close()

ax6 = fig.add_subplot(2,3,4)

print("Generating Figure 1")

for z in range(len(T_lambda)):
    ax6 = fig.add_subplot(2,3,4)
    #ax6.plot(T_T[z],T_lambda[z],marker="o",color=color[1],markersize=1)
    ax6.plot(T_T[z],T_lambda[z],marker="o",color=colstab[T_Stab[z]],markersize=2*T_Stab[z]+1) # add option to make marker size flexible
    plt.ylabel('lambda')
    plt.xlabel('T')
    plt.title('Tomahawk diagram (Black Stable)')

plt.show()

