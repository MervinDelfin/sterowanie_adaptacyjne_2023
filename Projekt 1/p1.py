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
            ysr[i] = None
        else:
            ysr[i] = np.sum(y[i-H:i]) / H
    return ysr

def MSE(y, yZ):
    return np.sum((y - yZ)**2) / len(y)


# Sygnał oryginalny, zakłócony oraz średnia ruchoma
# Rysujemy wykres
p1 = plt.figure(1)
plt.plot(t, y, label="Sygnał oryginalny")
plt.scatter(t, yZ, s=1, c='red', label='Sygnał zakłócony')
plt.xlabel('Czas')
plt.ylabel('Amplituda')
plt.title('Sygnał oryginalny, zakłócony oraz średnia ruchoma')

# Średnia ruchoma
H = 20
plt.plot(t, srednia_ruchoma(yZ, H), label="Średnia ruchoma H=20")


plt.legend(loc='upper right')
p1.show()
input()

# # Zad 1. Zależność MSE od H'
# def MSE_OD_H(y, yZ, H_max = 50):
#     y_MSE = np.zeros(H_max)
#     for H in range(1, H_max+1):
#         y_MSE[H-1] = MSE(y, srednia_ruchoma(yZ, H))
#     return y_MSE, np.argmin(y_MSE) + 1 # H_opt

# y_MSE, H_opt = MSE_OD_H(y, yZ)

# p2 = plt.figure(2)
# plt.plot(range(1, 50+1), y_MSE)
# plt.plot(H_opt, y_MSE[H_opt-1], 'ro', label='H_opt = ' + str(H_opt))
# plt.xlabel('H')
# plt.ylabel('MSE')
# plt.title('Zad 1. Zależność MSE od H')
# plt.legend(loc='upper right')
# p2.show()


# # Zad 2. Zależność H_opt od wariancji zakłócenia (średnia z 10 powtórzeń)
# p3 = plt.figure(3)
# MAX_Z_MULTIPLY = 5
# y_H_opt = np.zeros(MAX_Z_MULTIPLY)
# x_Var_Z = np.zeros(MAX_Z_MULTIPLY)
# REPEAT_COUNT = 10

# for ii, z_multiply in enumerate(range(1,MAX_Z_MULTIPLY+1)):
#     H_opts = 0
#     for iii in range(REPEAT_COUNT):
#         yZ = np.zeros(len(y))
#         for i in range(len(y)):
#             yZ[i] = y[i] + z_multiply * Z_trojkatne()
        
#         y_MSE, H_opt = MSE_OD_H(y, yZ)
#         H_opts += H_opt

#     y_H_opt[ii] = H_opts/REPEAT_COUNT
#     x_Var_Z[ii] = Z_trojkatne_wariancja(-1*z_multiply, 1*z_multiply)


# plt.scatter(x_Var_Z, y_H_opt)
# plt.xlabel('Var(Zk)')
# plt.ylabel('H_opt')
# plt.title('Zad 2. Zależność H_opt od Var(Zk)')
# p3.show()


# # Zad 3. Zależność MSE od wariancji zakłócenia
# p4 = plt.figure(4)
# y_MSE = np.zeros(MAX_Z_MULTIPLY)
# x_Var_Z = np.zeros(MAX_Z_MULTIPLY)
# for ii, z_multiply in enumerate(range(1,MAX_Z_MULTIPLY+1)):
#     yZ = np.zeros(len(y))
#     for i in range(len(y)):
#         yZ[i] = y[i] + Z_trojkatne(-1*z_multiply, 1*z_multiply)
    
#     y_MSE[ii] = MSE(y, yZ)
#     x_Var_Z[ii] = Z_trojkatne_wariancja(-1*z_multiply, 1*z_multiply)

    
# plt.scatter(x_Var_Z, y_MSE)
# plt.xlabel('Var(Zk)')
# plt.ylabel('MSE')
# plt.title('Zad 3. Zależność MSE od Var(Zk)')
# p4.show()

# input() # keep plots open