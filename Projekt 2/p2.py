import matplotlib.pyplot as plt
import numpy as np
import random, math

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


# ZAD 1 - system statyczny
def z1():
    # generacja sygnału
    b = [1.5, 0.8, 1.3]
    y_list = list()
    for i in range(1000):
        u = (random.random()*2)-1
        y = 0
        y_list.append((u,None))
        if len(y_list) < len(b):
            continue
        for ii in range(len(b)):
            y += b[ii]*y_list[-ii-1][0]

        y += Z_trojkatne(-0.2, 0.2)

        y_list[-1] = (u, y)

    return y_list

y_list = z1()

def identyfkacja(y_list, i, b, P, lambd=1):
    phi = np.array([y[0] for y in y_list[i-3+1:i+1][::-1]])

    # Calculate the predicted outputs
    y_pred = np.dot(b, phi)

    # Calculate the error
    e = y_list[i][1] - y_pred

    P = 1/lambd * (P - np.dot(P, np.dot(phi.T, np.dot(phi, P))) / (lambd + np.dot(phi, np.dot(P, phi.T))))

    # Calculate the gain vector
    K = np.dot(P, phi)

    # Update the parameter vector
    b = b + np.dot(K, e)

    return b, P

# Initialize the parameter vector b
b = np.array([1, 1, 1])

# Initialize the covariance matrix P
P = np.eye(len(b)) * 1000

# Iterate through the data points
for i in range(2, len(y_list)):
   b, P = identyfkacja(y_list, i, b, P)

# Print the identified parameters
print("1. Zidentyfikowane parametry b:", b)

# y = [y[1] for y in y_list]
# y = np.array(y)
# # t = [y[0] for y in y_list]
# # t = np.array(t)
# plt.scatter(t, y)
# plt.show()


# ZAD 2 - system dynamiczny
def z2():
    # generacja sygnału
    b = [1.5, 1, 1.3]
    y_list = list()

    # Definiujemy parametry fali
    amplitude = 0.2
    frequency = 0.008

    P = np.eye(len(b)) * 1000
    bi = np.array([1, 1, 1])

    u = 0

    for i in range(1000):

        ## SYMULACJA
        #zmiana b1 w czasie
        b[1] = 1 + amplitude * np.sign(np.sin(2 * np.pi * frequency * i))

        y = 0
        y_list.append((u,None))
        if len(y_list) < len(b):
            continue
        for ii in range(len(b)):
            y += b[ii]*y_list[-ii-1][0]

        y += Z_trojkatne(-0.2, 0.2)

        y_list[-1] = (u, y)

        ## IDENTYFIKACJA
        bi, P = identyfkacja(y_list, i, bi, P, lambd=0.9) # lambd - zapominanie (1 - nie zapomina, 0 - zapomina wszystko)

        ## Wyznaczenie u
        u = (1 - bi[1]*y_list[-1][0] - bi[2]*y_list[-2][0]) / bi[0]

    return y_list


y_list = z2()

y = [y[1] for y in y_list]
y = np.array(y)
# t = [y[0] for y in y_list]
# t = np.array(t)
t = np.linspace(0, 2, 1000)
plt.plot(t, y)
plt.show()
