    #TFM-Sonsoles
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
import scipy.optimize as optimize

    #Establecemos i = número imaginario

i   = np.complex(0.0,1.0)

    #Esta clase define puts y calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):


    # cf   - Función característica como una función, en el libro definida como \varphi
    # CP   - C para call y P para put
    # S0   - Precio de stock inicial
    # r    - Tipo de interés (constante)
    # tau  - Tiempo para madurar
    # K    - Lista de strikes
    # N    - Número de términos de expansión
    # L    - Tamaño del dominio de truncamiento (typ: L=8 o L=10)

    # Se convierte K en un vector columna

    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Se asigna i =sqrt(-1)

    i = np.complex(0.0,1.0) 
    x0 = np.log(S0 / K)   
    
    # Dominio de truncamiento

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Suma para k = 0 hasta k = N-1
    
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determinación de coeficientes para los precios put  

    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))     
    return value

    #Determinación de coeficientes para los precios put

def CallPutCoefficients(CP,a,b,k):
    if CP==OptionType.CALL:                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)  
    elif CP==OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k    

def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value
    
    #Precio de opciones call por Black - Scholes

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

    #Método de volatilidad implícita

def ImpliedVolatility(CP,marketPrice,K,T,S_0,r):

    # Para determinar la volatilidad inicial se define To determine initial volatility se define una cuadrícula para sigma
    # y se interpola en la función inversa

    sigmaGrid = np.linspace(0,2,200)
    optPriceGrid = BS_Call_Put_Option_Price(CP,S_0,K,sigmaGrid,T,r)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    
    # Utilizar la entrada ya determinada para la búsqueda local (ajuste final)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-10)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

def ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))

    # Se excluye el término -r*tau, debido a que el descuento se lleva a cabo en el método COS

    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    # Función característica del el modelo de Heston    

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf 

def mainCalculation():
    CP  = OptionType.CALL
    S0  = 100.0
    r   = 0.05
    tau = 2.0
    
    K = np.linspace(40,200,25)
    K = np.array(K).reshape([len(K),1])
    
    N = 500
    L = 5
    
    #Se defines los valores de los parámetros
        
    kappa = 0.1
    gamma = 0.1
    vbar  = 0.1
    rho   = -0.75
    v0    = 0.05
           
    # Efecto de gamma

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    gammaV = [0.1, 0.3, 0.5,0.9]
    legend = []
    for gammaTemp in gammaV:    

       # Evaluar el modelo de Heston
       # Ejecutar la ChF del modelo de Heston

       cf = ChFHestonModel(r,tau,kappa,gammaTemp,vbar,v0,rho)

       # Método COS

       valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)

       # Volatilidades implícitas

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatility(CP,valCOS[idx],K[idx],tau,S0,r)
       plt.plot(K,IV*100.0)
       legend.append('gamma={0}'.format(gammaTemp))
    plt.legend(legend)
    
    # Efecro de kappa
    
    plt.figure(2)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    kappaV = [0.1, 0.5, 1.0, 2.0]
    legend = []
    for kappaTemp in kappaV:    

       # Evaluar el modelo de Heston
       # Ejecutar la ChF del modelo de Heston

       cf = ChFHestonModel(r,tau,kappaTemp,gamma,vbar,v0,rho)

       # Método COS

       valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)      

       # Volatilidades implícitas

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatility(CP,valCOS[idx],K[idx],tau,S0,r)
       plt.plot(K,IV*100.0)
       legend.append('kappa={0}'.format(kappaTemp))
    plt.legend(legend)
    
    # Efecto de la dependencia del tiempo de kappa

    plt.figure(3)
    plt.grid()
    plt.xlabel('strike, K')
    plt.xlabel('time-to-mautrity T')
    kappaV = [0.1, 0.3, 0.5, 1.0, 2.0]
    legend = []
    Ktemp = [S0]
    maturityGrid = np.linspace(0.1,25,40)
    
    for kappaTemp in kappaV: 
        IV =np.zeros([len(maturityGrid),1])
        for idx, t in enumerate(maturityGrid):    

           # Evaluar el modelo de Heston
           # Ejecutar la ChF del modelo de Heston

           cf = ChFHestonModel(r,t,kappaTemp,gamma,vbar,v0,rho)

           # Método COS

           valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, t, Ktemp, N, L)

           # Volatilidades implícitas

           IV[idx] = ImpliedVolatility(CP,valCOS,Ktemp[0],t,S0,r)
        print(IV)   
        plt.plot(maturityGrid,IV*100.0)
        legend.append('kappa={0}'.format(kappaTemp))       
    plt.legend(legend)
    
    # Efecto de la dependencia del tiempo en v_0

    plt.figure(4)
    plt.grid()
    plt.xlabel('time-to-mautrity T')
    plt.ylabel('implied volatility')
    v0V = [0.1, 0.3, 0.5, 1.0, 2.0]
    legend = []
    Ktemp = [S0]
    maturityGrid = np.linspace(0.1,25,40)
    
    for v0Temp in v0V: 
        IV =np.zeros([len(maturityGrid),1])
        for idx, t in enumerate(maturityGrid):    

           # Evaluar el modelo de Heston
           # Ejecutar la ChF del modelo de Heston

           cf = ChFHestonModel(r,t,kappa,gamma,vbar,v0Temp,rho)

           # Método COS

           valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, t, Ktemp, N, L)

           # Volatilidades implícitas

           IV[idx] = ImpliedVolatility(CP,valCOS,Ktemp[0],t,S0,r)
        plt.plot(maturityGrid,IV*100.0)
        legend.append('v0={0}'.format(v0Temp))       
    plt.legend(legend)
      
    # Efecto de rho

    plt.figure(5)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    rhoV = [0.0, -0.25, -0.5,  -0.9]
    legend = []
    for rhoTemp in rhoV:    

       # Evaluar el modelo de Heston
       # Ejecutar la ChF del modelo de Heston

       cf = ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rhoTemp)
         
       # Método COS

       valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)
        
       # Volatilidades implícitas

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatility(CP,valCOS[idx],K[idx],tau,S0,r)
       plt.plot(K,IV*100.0)
       #plt.plot(K,valCOS)
       legend.append('rho={0}'.format(rhoTemp))
    plt.legend(legend)
    
    # Efecto de vbar

    plt.figure(6)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    vbarV = [0.03, 0.1, 0.2, 0.3]
    legend = []
    for vbarTemp in vbarV:    

       # Evaluar el modelo de Heston
       # Ejecutar la ChF del modelo de Heston

       cf = ChFHestonModel(r,tau,kappa,gamma,vbarTemp,v0,rho)
         
       # Método COS

       valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)
        
       # Volatilidades implícitas

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatility(CP,valCOS[idx],K[idx],tau,S0,r)
       plt.plot(K,IV*100.0)
       legend.append('vbar={0}'.format(vbarTemp))
    plt.legend(legend)
    
mainCalculation()
