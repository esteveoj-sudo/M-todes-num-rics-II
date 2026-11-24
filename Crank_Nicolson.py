import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Anomenem les constants:
d = 0.02
v = 40
cv = 3698
p = 1081
k = 0.56
s = 0.472
p_o = (v**2*s)/(2*d**2)
ap = (k)/(cv*p) #Constant alpha
b = p_o*d**2/k #Constant beta. Per tant la EDP queda com: (dT/dt) = (d^2T/dx^2) + 1
#Posem valors als mallats:
x = np.linspace(0,1,101) #Elements equiespaiats de 0 a 2 (D'un electròde a un altre)
dx = 1/100
#Mètode Crank-Nicolson:
def crank_nicolson(h,t_a):
    """
    Calcula la temperatura en el domini establert seguida per l'equació 
    dT/dt = a*(d^2T/dx^2) + b segons el mètode de crank-nicolson

    Args:
        h(float): Relació mallat temporal amb l'espacial
        t_a(int): temps final de càlcul
        cond: Volem aplicar els requisits del problema? Sí = 1; No = 0
    Els requisits del problema que necesitem són:
        • Que la regió sana no arribi a 50ºC.
        • Que cap punt arribi a 80ºC.
    Returns:
        matriu: Valor de les temperatures a t_a per a cada punt x_idel domini  
    """
    dt = h*dx**2 #Salt entre temps (S'aplica la definició que esta a l'enunciat)
    e = int(t_a/dt +1) #N del mallat temporal
    t = np.linspace(0,t_a,e) #Mallat temporal
    a = h/2 #Valor que ens servira per a les matrius del mètode Gauss-Seidel
    #Valors de T:
    tp =np.zeros((101,e)) #Temperatura en les 2 variables, a cada fila s'ha assignat una x fixada i a cada columna una t fixada. Per tant tp[i,j] serà T(xi,tj)
    '''
    És una matriu plena de zeros degut a que la normalització de la condició de contorn donada per a la temperatura és 0.
    Els punts que no formen part de la condició de contorn es modificaran en els següents passos
    '''
    #Creem les matrius que utilitzarem per resoldre el sistema amb Gauss-Seidel:
    #Matriu G:
    g = np.zeros((101,101)) 
    g[0,0] = 1 #Aixo es degut a les condicions de contorn, q no es modifiquen els elements de les files 0 i 100
    g[100,100] = 1 #Cond de contorn
    for i in range(1,100):
        g[i,i-1:i+1] = [-a,1+2*a] #El q va a cada fila i cada columna
    invg = np.linalg.inv(g)
    #Matriu S
    s = np.zeros((101,101))
    for i in range(1,100):
        s[i,i+1] = -a #El q va a cada fila i columna(a a primera fila i a la ultima )
    #Matriu N
    n = np.zeros((101,101))
    n[0,0] = 0
    n[100,100] = 0
    for i in range(1,100):
            n[i,i-1:i+2] = [a,1-2*a,a]
    #Matriu T
    t_m = dt*np.ones((101,1))
    t_m[0,0] = 0
    t_m[100,0]= 0
    #Comencem a iterar:
    for i in range(1,e):
        k_m = np.zeros((101,2)) #Per començar creem una matriu de 101 files i 2 columnes, ens servirà per a tenir els valors de T en el punt que estem tractant i en l'anterior
        k_m[:,0] = tp[:,i-1] #Per a començar el mètode s'otorga el valor T^n a T^k
        ntpt = (n@tp)[:,i-1] + t_m[:,0] #Es el valor de GT^n +T
        error  = 1 #Donem un valor a l'error per a que el bucle del mètode comenci:
        while error > 1e-15:
            p = invg@(ntpt - s@k_m[:,0])
            k_m[:,1] = p
            error = np.max(np.abs(k_m[:,1]-k_m[:,0]))
            k_m[:,0] = k_m[:,1] #Al haver calculat ja la iteració podem suprimir la anterior
        tp[1:100,i] = k_m[1:100,0] #Si la condició d'error es compleix, confirmem el valor i ho posem a la matriu final de Temperatures
        m = i #Ens proporciona les iteracions totals realitzades
    temperatura = tp[:,-1]
    return (temperatura*b+36.5*np.ones(101))  #Ens dona les temperatures (en ºC) en el domini x = [0,0.02], t = t_a
#Mètode Analític:
def analitic(t_a,n):
    """
        Calcula la temperatura en el domini establert seguida per l'equació 
        dT/dt = a*(d^2T/dx^2) + b segons una resolució analítica per un temps fixat
    
        Args:
            t_a(float): temps de càlcul
            n (int): Nombre d'iteracions realitzades, determinen la precisió de la solució analítica
    
        Return: Temperatura calculada analíticament en un temps fixat(t_a) en el domini 0<=x<=1
        """
    temp_analitica = np.full(101,0.0) #Començem amb el valor 0
    for i in range(1,n, 2): #Creem un bucle 'for' que faci n iteracions per arribar al resultat
        temp_analitica += 4/(i**3*np.pi**3)*(1-np.e**(-i**2*np.pi**2*t_a))*np.sin(i*np.pi*x) #És simplement aplicar la fórmula de la solució
    return temp_analitica*b + 36.5*np.full(101,1)
#Comparació Mètodes:
t_a = 0.025
#1 solució analítica:
sol_analitica = analitic(t_a,1000)
#2 Crank-nicolson h = 1
cn_1 = crank_nicolson(1,t_a)
error_cn1 = np.abs(cn_1 - sol_analitica)
#3 Crank-nicolson h = 0.5
cn_2 = crank_nicolson(0.5,t_a)
error_cn2 = np.abs(cn_2 - sol_analitica)
x_2 = 2*x #Així tenim un eix en cm
plt.plot(x_2,sol_analitica, label = 'Solució Analítica', color = 'black', linewidth=2)
plt.plot(x_2,cn_1, label = r'Crank-Nicolson $\gamma = 1$', color ='orange', linestyle = '--')
plt.plot(x_2,cn_2, label = r'Crank-Nicolson $\gamma = 0.5$', color ='green', linestyle = ':')
plt.title(f'Temperatura per crank-nicolson a t = {t_a}')
plt.xlabel("x (cm)")
plt.ylabel("T(ºC)")
plt.grid(True)
plt.legend()
plt.show()
plt.plot(x_2,error_cn1,label = r'$\gamma =1$', color = 'orange')
plt.plot(x_2,error_cn2, label = r'$\gamma = 0.5$', color = 'green')
plt.title("Errors del mètode Crank-Nicolson")
plt.xlabel("x (cm)")
plt.ylabel("Error")
plt.grid(True)
plt.legend()
plt.show()
