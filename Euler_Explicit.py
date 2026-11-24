import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.ticker import MultipleLocator

#Determinació de les variables i llistes que aniré usant en el codi. També la llista amb les temperatures a temps inicial.
T_0 = [0]*101
gamma1 = 0.25
gamma2 = 0.49
gamma3 = 0.51

dx = 0.0002
dt = gamma1*(dx**2)

l = 0.02
T0 = 674.285714     
t0 = 2855.384286   

#Definició d'una funció que genera una llista T amb subllistes T_n, on les subllistes T_n són les temperatures a cada punt de la discretització en el temps n.
def dif(T_0, gamma, N, t_lim):
    """
    T_0 --> list de condicions inicials 
    gamma --> float on es determina la gamma amb la qual s'opera
    N --> int que explicita el nombre de punts de la discretització
    t_lim --> float on es determina fins a quin temps (normalitzat) es vol calcular
    """
    T = [T_0]
    dt = gamma*(1/(N-1)**2)
    a =  int(t_lim/dt)
    rang = range(len(T_0))[1:-1]
    for j in range(a):
        T_n = [0]
        T_n_ant = T[j]
        for i in rang:
            Tn = T_n_ant[i] + gamma*(T_n_ant[i+1] - 2*T_n_ant[i] + T_n_ant[i-1]) + dt
            T_n.append(Tn)
        T_n.append(0)
        T.append(T_n)
    return T

#Generació de les "matrius" de temperatura en els diversos punts del matllat espai-temporal amb les gammes demanades i fins a t_a. 
T25 = dif(T_0, gamma1, 101, 0.025)
T49 = dif(T_0, gamma2, 101, 0.025)
T51 = dif(T_0, gamma3, 101, 0.025)

#Es crea una llista per de temperatures desnormalitzades a l'últim temps de les llistes anteriors.
T25f = [36.5+i*674.285714 for i in T25[-1]]
T49f = [36.5+i*674.285714 for i in T49[-1]]
T51f = [36.5+i*674.285714 for i in T51[-1]]

#Es desnormalitza tot T25
T252t = []
for i in T25:
    T252f = [36.5+j*674.285714 for j in i]
    T252t.append(T252f)

#Es genera una llista amb cada posició
X51 = [i/100*0.02 for i in range(len(T51[-1]))]

#Definició d'una funció que genera una aproximació dels N primers termes de la solució anlaítica de l'equació de difusió normalitzada.
def sigma(N, ta, x):
    """
    N --> int on s'especifíca els termes que es volen de l'aproximació
    ta --> float on es determina fins a quin temps es vol fer l'aproximació
    x --> list de totes les posicións
    """
    sigma = []
    for x_el in x:
        suma = 0.0
        for k in range(N):
            n = 2*k + 1
            suma += 4*(1-np.exp(-n**2*np.pi**2*ta))/(n**3*np.pi**3)*np.sin(n*np.pi*x_el)
        sigma.append(suma)
    return sigma

ta = 0.025
x_n = np.linspace(0,1,101)


#Llista desnormalitzada de totes les temperatures "reals" en el temps t_a.
T_r = [36.5 + T0 * float(i)  for i in sigma(1000, ta, x_n)]

#Llistes dels errors en valor absolut per a cada posició
e25 = [np.abs(a - b) for a, b in zip(T_r, T25f)]
e49 = [np.abs(a - b) for a, b in zip(T_r, T49f)]
e51 = [np.abs(a - b) for a, b in zip(T_r, T51f)]

#Aplicant les condicions del problema a la llista T252t per trobar el temps final i les seves temperatures.
a = False
b = 1
for i in T252t:
    dt = gamma1*(1/101**2)
    b = T252t.index(i)
    for j in i:
        if (j > 50 and i.index(j) < 37) or (j>80):
            print(f"El temps màxim és de {b*dt*t0} s")
            a = True
            break
    if a:
        break

#Graficació dels resultats obtinguts.
fig1, ax = plt.subplots(2, 3, figsize=(8, 10))

ax[0,0].plot(2*x_n, T25f, color='blue', label=f'gamma = {gamma1}')
ax[0,0].set_title(r"$\gamma$ = 0.25")
ax[0,0].set_xlabel("Posició X (cm)")
ax[0,0].set_ylabel("Temperatura T (ºC)")
ax[0,0].grid(True)

ax[0,1].plot(2*x_n, T49f, color='orange', label=f'gamma = {gamma2}')
ax[0,1].set_title(r"$\gamma$ = 0.49")
ax[0,1].set_xlabel("Posició X (cm)")
ax[0,1].set_ylabel("Temperatura T (ºC)")
ax[0,1].grid(True)

ax[0,2].plot(2*x_n, T51f, color='red', label=f'gamma = {gamma3}')
ax[0,2].set_title(r"$\gamma$ = 0.51")
ax[0,2].set_xlabel("Posició X (cm)")
ax[0,2].set_ylabel("Temperatura (ºC)")
ax[0,2].set_yscale("symlog")
ax[0,2].grid(True, which="both", ls="--")

ax[1,0].plot(2*x_n, e25, color='blue', label=f'gamma = {gamma1}')
ax[1,0].set_title(r"$\gamma$  = 0.25")
ax[1,0].set_xlabel("Posició X (cm)")
ax[1,0].set_ylabel("Error")
ax[1,0].grid(True)

ax[1,1].plot(2*x_n, e49, color='orange', label=f'gamma = {gamma2}')
ax[1,1].set_title(r"$\gamma$ = 0.49")
ax[1,1].set_xlabel("Posició X (cm)")
ax[1,1].set_ylabel("Error")
ax[1,1].grid(True)

ax[1,2].plot(2*x_n, e51, color='red', label=f'gamma$ = {gamma3}')
ax[1,2].set_title(r"$\gamma$ = 0.51")
ax[1,2].set_xlabel("Posició X (cm)")
ax[1,2].set_ylabel("Error")
ax[1,2].set_yscale("symlog")
ax[1,2].grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()

fig3, ix1 = plt.subplots(1,1, figsize=(8,10))

x_centre = 1
amplada = 0.5
x_min_zona = x_centre - amplada / 2
x_max_zona = x_centre + amplada / 2

ix1.plot(2*x_n,T252t[b])
ix1.axvspan(x_min_zona, x_max_zona, color='blue', alpha=0.3, label='Teixit malalt (0,5 cm)')
ix1.axvspan(0, x_min_zona, color='red', alpha=0.3)
ix1.axvspan(x_max_zona, 2, color='red', alpha=0.3, label='Teixit sa')
ix1.axvline(x_min_zona, color='goldenrod', linestyle='--', linewidth=1)
ix1.axvline(x_max_zona, color='goldenrod', linestyle='--', linewidth=1)
ix1.axhline(50, color='gray', linestyle=':', linewidth=1, label='Llindar de 50 °C')
ix1.axhline(80, color='red', linestyle=':', linewidth=1, label='Llindar de 80 °C')
ix1.set_xlabel("Posició X (m)")
ix1.set_ylabel("Temperatura T (°C)")

plt.tight_layout()
plt.show()


