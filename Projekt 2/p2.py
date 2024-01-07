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

        # y += Z_trojkatne(-0.2, 0.2)

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

b0= list()
b1= list()
b2= list()

# Iterate through the data points
for i in range(2, len(y_list)):
    b, P = identyfkacja(y_list, i, b, P)
    b0.append(b[0])
    b1.append(b[1])
    b2.append(b[2])

# Print the identified parameters
print("1. Zidentyfikowane parametry b:", b)

# wykres parametrow b0, b1, b2 w czasie
# t = np.linspace(0, 998, 998)
# plt.plot(t, b0, label="b0")
# plt.plot(t, b1, label="b1")
# plt.plot(t, b2, label="b2")
# plt.legend()
# plt.show()


# y = [y[1] for y in y_list]
# y = np.array(y)
# # t = [y[0] for y in y_list]
# # t = np.array(t)
# plt.scatter(t, y)
# plt.show()

# ZAD 2 - system dynamiczny
def z2(lambd=0.95):
    global b0,b1,b2, b0_real,b1_real,b2_real

    b0= list()
    b1= list()
    b2= list()

    b0_real = list()
    b1_real = list()
    b2_real = list()

    # generacja sygnału
    b = [1.5, 1, 1.3]
    y_list = list()

    # Definiujemy parametry fali
    amplitude = 0.2
    frequency = 0.0008

    P = np.eye(len(b)) * 1000
    bi = np.array([1, 1, 1])

    u = 0

    for i in range(1000):

        ## SYMULACJA
        #zmiana b1 w czasie
        # b[1] = 1 + amplitude * np.sign(np.sin(2 * np.pi * frequency * i))
        b[1] = 1 + amplitude * np.sin(2 * np.pi * frequency * i)


        b0_real.append(b[0])
        b1_real.append(b[1])
        b2_real.append(b[2])

        y = 0
        y_list.append((u,None))
        if len(y_list) < len(b):
            b0.append(None)
            b1.append(None)
            b2.append(None)
            continue

        for ii in range(len(b)):
            y += b[ii]*y_list[-ii-1][0]

        # y += Z_trojkatne(-0.2, 0.2)

        y_list[-1] = (u, y)

        ## IDENTYFIKACJA
        bi, P = identyfkacja(y_list, i, bi, P, lambd=lambd) # lambd - zapominanie (1 - nie zapomina, 0 - zapomina wszystko)
        b0.append(bi[0])
        b1.append(bi[1])
        b2.append(bi[2])

        ## Wyznaczenie u
        # if i >= 500:
        #     u = (1 - bi[1]*y_list[-1][0] - bi[2]*y_list[-2][0]) / bi[0]
        # else:
        u = (random.random()*2)-1
        

    return y_list


y_list = z2()

# y = [y[1] for y in y_list]
# y = np.array(y)
# # t = [y[0] for y in y_list]
# # t = np.array(t)
# t = np.linspace(0, 2, 1000)
# plt.plot(t, y)
# plt.show()


# wykres parametrow b0, b1, b2 w czasie
t = np.linspace(0, 1000, 1000)
plt.plot(t, b0, label="b0", color="r")
plt.plot(t, b1, label="b1", color="b")
plt.plot(t, b2, label="b2", color="g")
plt.plot(t, b0_real, color="r", linestyle="--")
plt.plot(t, b1_real, color="b", linestyle="--")
plt.plot(t, b2_real, color="g", linestyle="--")
plt.plot(t, [y[0] for y in y_list], label="u")
# plt.plot(t, [y[1] for y in y_list], label="y")
plt.legend(loc='lower right')  # Set the legend position to lower right

plt.show()



#MSE w zależności od lambda
mse_lambda = np.arange(0.01,1.0,0.01)
mse = np.zeros(len(mse_lambda))

for i, l in enumerate(mse_lambda):
    y_list = z2(l)
    mse[i] = sum([(b1[ii]-b1_real[ii])**2 for ii in range(len(b1)) if b1[ii]!=None])/99

plt.plot(mse_lambda, mse)
plt.show()