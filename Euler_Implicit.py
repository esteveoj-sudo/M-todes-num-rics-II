import numpy as np
import matplotlib.pyplot as plt
# Anomenem les constants:
d = 0.02
v = 40
cv = 3698
p = 1081
k = 0.56
s = 0.472
p_o = (v**2*s)/(2*d**2)
ap = (k)/(cv*p) # Constant alpha
b = p_o*d**2/k # Constant beta. Per tant la EDP queda com: (dT/dt) = (d^2T/dx^2) + 1
T_0 = b

def euler_implicit(h, dx, T_c, N, t_a):
    # Calculem magnituds que farem anar més endavant (que depenen dels paràmetres que nosaltres decidim introduir)
    dt = h*dx**2
    punts_t = int(t_a/dt + 1)
    gamma = dt/dx**2
    a = 1 + 2*gamma

    # T_c en graus Kelvin
    T_cK = T_c + 273.15

    # Normalitzem T_c (temperatura corporal que hem introduït)
    Tc = (T_c - T_c)/T_0

    # L'equació a resoldre és AT^{i} = T^{i-1} + Dt
    # Aïllem per a cada iteració del mètode (C+D)T^{i}_{k+1} = T^{i-1} + Dt - ET^{i}_{k} per a la k iteració
    # Separem A en A = C + D + E
    C = np.zeros((N-2,N-2))
    D = np.zeros((N-2,N-2))
    E = np.zeros((N-2,N-2))
    for i in range(N-3):
        E[i,i+1] = E[i,i+1] - gamma
    for i in range(N-2):
        D[i,i] = D[i,i] + a
    for i in range(N-3):
        C[i+1,i] = C[i+1,i] - gamma

    # Condició inicial per a la primera iteració del primer sistema d'equacions. Obviem els punts dels extrems, ja que no calen per al bucle de després
    T_k = np.full(N-2,Tc) 

    # Vector Dt que forma part de l'equació original
    Dt = np.full(N-4, dt)

    # El modifiquem per a incloure-hi els extrems
    Dt = np.append(Dt,0)
    Dt = np.insert(Dt, 0, 0)

    # Matriu R (resultats) per a emmagatzemar els resultats 
    R = np.full(N, Tc).reshape(-1,1) # Amb el comandament reshape fem que sigui un vector columna

    # Amb totes les components, fem 3 bucles; un per a cada temps, un altre per a cada iteració del mètode i un altre per resoldre el sistema amb Gauss-Seidel
    for i in range(1, punts_t): # Per a cada temps (el primer està fixat per les condicions inicials)
        max_iter = 100000 # Introduïm un comptador màxim d'iteracions per si de cas
        iter_count = 0
        while True:
            iter_count += 1
            # Càlcul de r_k
            r_k = R[1:-1,i-1] + Dt - E @ T_k # Prenem els elements de la columna i-1 (t = i-1) i obviem els elements de les condicions de contorn per a fer el càlcul
            # Càlcul de T_{k+1} = T_k1 (nomenclatura)
            T_k1 = np.zeros(N-2) # Comencem amb un vector horitzontal de 99 elements (són 0)
            for j in range(len(r_k)): # Fem un bucle per a resoldre el sistema (C+D)T_k1 = r_k
                suma = 0.0 
                for k in range(j): # Fem un bucle per a calcular les components que no són la variable que estem trobant (component)
                    suma = suma+(C[j, k]+D[j, k])*T_k1[k]
                T_k1[j] = (r_k[j]-suma)/(C[j, j]+D[j, j]) # Calculem tot el vector T_k1
            dif = np.max(np.abs(T_k1 - T_k)) # Definim la diferència entre iteracions per a saber quan aturar-ho
            T_k = T_k1.copy() # Guardem el valor de T_k1 per a la següent iteració (substituïnt T_k)
            if dif < 1e-15 or iter_count > max_iter: # Imposem dues condicions per a aturar el bucle (per si la solució no convergís)
                T_k1 = np.append(T_k1,Tc) # Afegim la condició de contorn a un dels extrems
                T_k1 = np.insert(T_k1,0,Tc) # L'afegim també a l'altre extrem
                break
            # Cal, a més a més, que tinguem en compte les temperatures del teixit sa i del teixit malalt, ja que no poden excedir certs valors
        R = np.hstack((R, T_k1.reshape(-1,1))) # Afegim la columna de resultats a la matriu R per a anar actualitzant els resultats
    R = (T_cK + T_0*R) - 273.15 # En Celsius
    return R # Acabem retornant la matriu amb tots els valors per a cada x i t

# Definim la solució analítica per a fer comparacions
def analitic(t_a,n):
    temp_analitica = np.full(101,0.0) # Començem amb el valor 0
    for i in range(1,n, 2): # Creem un bucle que faci n iteracions per a arribar al resultat
        temp_analitica += 4/(i**3*np.pi**3)*(1-np.e**(-i**2*np.pi**2*t_a))*np.sin(i*np.pi*x) # És simplement aplicar la fórmula de la solució
    return temp_analitica*b + 36.5*np.full(101,1)

# Comparació d'ambdós mètodes, analític i per Euler Implícit
x = np.linspace(0,1,101) #Elements equiespaiats de 0 a 2 (D'un electròde a un altre)
t_a = 0.025
h1 = 1
h2 = 1/2
dx = 1/100
T_c = 36.5
N = 101

# Calculem les temperatures a tot l'espai a t = 0,025 (normalitzat)
temperatures_t_a_1 = euler_implicit(h1, dx, T_c, N, t_a)[:, -1]
temperatures_t_a_2 = euler_implicit(h2, dx, T_c, N, t_a)[:, -1]

# Calculem la solució analítica
sol_analitica = analitic(t_a,10000)

# Calculem els errors
e1 = np.abs(temperatures_t_a_1 - sol_analitica)
e2 = np.abs(temperatures_t_a_2 - sol_analitica)


fig1, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

# Gràfica 1: gamma = 1
ax[0, 0].plot(2*x, temperatures_t_a_1, color='blue', label=r'$\gamma = 1$')
ax[0, 0].set_title(r"$\gamma = dt/(dx^2)$")
ax[0, 0].set_xlabel("Posició X (cm)")
ax[0, 0].set_ylabel("Temperatura T (ºC)")
ax[0, 0].grid(True)
ax[0, 0].legend()

# Gràfica 2: gamma = 1/2
ax[0, 1].plot(2*x, temperatures_t_a_2, color='orange', label=r'$\gamma = 1/2$')
ax[0, 1].set_title(r"$\gamma = dt/(2(dx)^2)$")
ax[0, 1].set_xlabel("Posició X (cm)")
ax[0, 1].set_ylabel("Temperatura T (ºC)")
ax[0, 1].grid(True)
ax[0, 1].legend()

# Gràfica 3: error gamma = 1
ax[1, 0].plot(2*x, e1, color='blue', label=r'$\gamma = 1$')
ax[1, 0].set_title(r"$\gamma = dt/(dx^2)$")
ax[1, 0].set_xlabel("Posició X (cm)")
ax[1, 0].set_ylabel("Error")
ax[1, 0].grid(True)
ax[1, 0].legend()

# Gràfica 4: error gamma = 1/2
ax[1, 1].plot(2*x, e2, color='orange', label=r'$\gamma = 1/2$')
ax[1, 1].set_title(r"$\gamma = dt/(2(dx)^2)$")
ax[1, 1].set_xlabel("Posició X (cm)")
ax[1, 1].set_ylabel("Error")
ax[1, 1].grid(True)
ax[1, 1].legend()

plt.show()