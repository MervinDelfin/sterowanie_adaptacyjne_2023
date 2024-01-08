import matplotlib.pyplot as plt
import numpy as np
import random, math
from scipy.optimize import fmin

# zakłócenie trójkątne
# https://pl.wikipedia.org/wiki/Rozkład_trójkątny
def Z_trojkatne(a=-1, b=1):
    if a*a - a*b == 0:
        raise TypeError("a^2 - a*b == 0")
    if a*b - b*b == 0:
        raise TypeError("a*b - b^2 == 0")
    
    u = random.random()
    if u <= 0.5:
        return a + math.sqrt(a*u*(a-b))
    else:
        return b - math.sqrt(-b*(u-1)*(b-a))

def Z_trojkatne_wariancja(a=-1, b=1):
    return (a*a + b*b - a*b) / 18


H = np.array([[0, 1], 
                [1, 0]])
    
def symulator():
    U = np.array([[random.random()] for _ in range(2)])
        
    A = np.array([[0.5, 0], 
                  [0, 0.25]])
    B = np.array([[1, 0], 
                  [0, 1]])
    Z = np.array([[Z_trojkatne(-0.2, 0.2)] for _ in range(2)])
    #Z = np.array([[0], [0]])

    Y = np.linalg.inv(np.identity(2) - A@H)@B@U + np.linalg.inv(np.identity(2) - A@H)@Z

    return U, Y


def identyfikacja():
    Ylist = [[] for _ in range(2)]
    Wlist = [[] for _ in range(2)]
    for _ in range(10000):
        U, Y = symulator()
        
        for ii in range(2):
            Ylist[ii].append(Y[ii])
            # print(np.dot(H,Y)[ii])
            Wlist[ii].append([
                U[ii][0],
                np.dot(H,Y)[ii][0]
            ])

    Ylist = np.array(Ylist)
    Wlist = np.array(Wlist)
    A, B = [0,0], [0,0]
    for ii in range(2):
        A[ii], B[ii] = (np.dot(Ylist[ii].T , Wlist[ii]) @ np.linalg.inv(Wlist[ii].T @ Wlist[ii]))[0]

    return A, B

A,B = identyfikacja()
print('A =', A)
print('B =', B)


# Sterowanie

A = np.array([
    [A[0], 0],
    [0, A[1]]
])
B = np.array([
    [B[0], 0],
    [0, B[1]]
])

def obiekt(U, A, B):
    if U.shape != (2,1):
        raise TypeError("U.shape != (2,1)")
    if A.shape != (2,2):
        raise TypeError("A.shape != (2,2)")
    if B.shape != (2,2):
        raise TypeError("B.shape != (2,2)")

    Y = np.linalg.inv(np.identity(2) - A@H)@B@U

    return U, Y

def Q(u1, u2, A, B, none=False):

    #ograniczenie 
    if u1[0]**2 + u2[0]**2 <= 1:
        if none:
            return None
        return 99999999 - 1000* (u1[0]**2 + u2[0]**2)

    U, Y = obiekt(np.array([u1, u2]), A, B)

    return (Y[1][0] - 4)**2 + (Y[0][0] - 4)**2
    

def Q1(u2, A, B):
    u1opt = fmin(Q, [0], args=(u2,A,B), maxiter=1000, disp=False)
    return Q(u1opt, u2, A, B)

#Metoda Neldera-Meada
u2opt = fmin(Q1, [0], args=(A,B), maxiter=1000)[0]
u1opt = fmin(Q, [5], args=([u2opt],A,B), maxiter=1000, disp=False)[0]
print("u1opt =", u1opt)
print("u2opt =", u2opt)
print("Q min =", Q([u1opt], [u2opt], A, B))

U, Y = obiekt(np.array([[u1opt], [u2opt]]), A, B)
print("Y =", Y)


def fig1():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    NUM_POINTS = 100
    u1 = np.linspace(-4, 4, NUM_POINTS)
    u2 = np.linspace(-4, 4, NUM_POINTS)
    u1, u2 = np.meshgrid(u1, u2)
    Qlist = np.array([[Q([u1[ii, jj]], [u2[ii, jj]], A, B, none=True) for ii in range(NUM_POINTS)] for jj in range(NUM_POINTS)])

    ax.plot_surface(u1, u2, Qlist, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.scatter(u1opt, u2opt, Q([u1opt], [u2opt], A, B), c='r', marker='o')
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('Q')
    plt.show()


def fig2():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    NUM_POINTS = 100
    u1 = np.linspace(u1opt-0.001, u1opt+0.001, NUM_POINTS)
    u2 = np.linspace(u2opt-0.001, u2opt+0.001, NUM_POINTS)
    u1, u2 = np.meshgrid(u1, u2)
    Qlist = np.array([[Q([u1[ii, jj]], [u2[ii, jj]], A, B, none=True) for ii in range(NUM_POINTS)] for jj in range(NUM_POINTS)])

    ax.plot_surface(u1, u2, Qlist, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.scatter(u1opt, u2opt, Q([u1opt], [u2opt], A, B), c='r', marker='o')
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('Q')
    plt.show()

fig1()
fig2()