import matplotlib.pyplot as plt
import numpy as np
import random, math

# Definiujemy parametry fali
amplitude = 1
frequency = 1
phase = 0
duration = 2

# Tworzymy os czasu
t = np.linspace(0, duration, 1000)

# Obliczamy wartości fali
y = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase))

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



def z1():
    # generacja sygnału
    b = [1.5, 0.8, 1.3]
    y_list = list()
    for i in range(10000):
        u = (random.random()*2)-1
        y = 0
        y_list.append((u,None))
        if len(y_list) < len(b):
            continue
        for ii in range(len(b)):
            y += b[ii]*y_list[-ii-1][0]

        y += Z_trojkatne(-0.3, 0.3)

        y_list[-1] = (u, y)

    # dentyfikacja parametów b*
    theta = np.ones(len(b))
    P = np.eye(len(b), dtype='int') * 1000 # macierz kowariancji

    for i in range(len(y_list)):
        if y_list[i][1] == None:
            continue
        x = np.array([y_list[i-ii-1][0] for ii in range(len(b))])
        x = np.flip(x)
        x = np.array([x])
        P = P - (P @ x.T @ x @ P) / (1 + x @ P @ x.T)
        theta = theta + P @ x.T * (y_list[i][1] - x @ theta.T)
    print(theta)

    return y_list

l = z1()
y_list = [y[1] for y in l]
y_list = np.array(y_list)
t = [y[0] for y in l]
t = np.array(t)
plt.scatter(t, y_list)
plt.show()
