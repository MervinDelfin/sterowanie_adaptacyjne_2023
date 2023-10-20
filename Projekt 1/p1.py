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
def Z_trojkatne():
    a = -1
    b = 1
    if a*a - a*b == 0:
        raise TypeError("a^2 - a*b == 0")
    if a*b - b*b == 0:
        raise TypeError("a*b - b^2 == 0")
    
    u = random.random()
    if u <= 0.5:
        return a + math.sqrt(a*u*(a-b))
    else:
        return b - math.sqrt(-b*(u-1)*(b-a))


### Test zakłócenia trójkątnego
# t2 = np.linspace(-1, 1, 1000)
# y2 = np.zeros(len(t2))
# for i in range(100000):
#     zz = z()
#     zz = round(((zz/2)+0.5)*1000)
#     print(zz)
#     y2[zz] += 1
# plt.plot(t2, y2)

yZ = np.zeros(len(y))
for i in range(len(y)):
    yZ[i] = y[i] + Z_trojkatne()

# Średnia ruchoma
def srednia_ruchoma(y, H):
    ysr = np.zeros(len(y))
    for i in range(len(y)):
        if i < H:
            ysr[i] = np.sum(y[0:i]) / i
        else:
            ysr[i] = np.sum(y[i-H:i]) / H
    return ysr

# Rysujemy wykres
plt.plot(t, y)
plt.plot(t, yZ)
plt.xlabel('Czas')
plt.ylabel('Amplituda')
plt.show()
