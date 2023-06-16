#%% H1-HW model COS method

import enum 
import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
from scipy.optimize import minimize


# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, tau, K, N, L, P0T):
    # cf   - Characteristic function as a functon, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)  
    # P0T  - Zero-coupon bond for maturity T.
        
    # Reshape K to become a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Assigning i=sqrt(-1)
    i = complex(0.0,1.0) 
    x0 = np.log(S0 / K)   
    
    # Truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Summation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a)  

    # Determine coefficients for put prices  
    H_k = CallPutCoefficients(OptionType.PUT,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = K * np.real(mat.dot(temp))     
    
    # We use the put-call parity for call options
    if CP == OptionType.CALL:
        value = value + S0 - K * P0T
        
    return value

# Determine coefficients for put prices 
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
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)
    return H_k    

def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - \
        np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d) - \
        np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - \
        k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

# Exact expectation E[sqrt(V(t))]
def meanSqrtV_3(kappa, v0, vbar, gamma):
    delta = 4.0 * kappa * vbar/gamma/gamma
    c = lambda t: 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t)))
    kappaBar = lambda t: 4.0*kappa*v0*np.exp(-kappa*t)/(gamma*gamma*(1.0- \
            np.exp(-kappa*t)))
    temp1 = lambda t: np.sqrt(2.0*c(t))*sp.gamma((1.0 + \
        delta)/2.0)/sp.gamma(delta/2.0)*sp.hyp1f1(-0.5,delta/2.0,-kappaBar(t)/2.0)
    return temp1
    
def A_H1HW(u,tau,P0T,lambd,eta,kappa,gamma,vbar,v0,rhoxv,rhoxr):
    i  = complex(0.0,1.0)
    D1 = np.sqrt(np.power(kappa-gamma*rhoxv*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = (kappa-gamma*rhoxv*i*u-D1)/(kappa-gamma*rhoxv*i*u+D1)
    
    # Function theta(t)
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + \
        eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))  

    # Integration within the function I_1
    N  = 500
    z  = np.linspace(0,tau-1e-10,N)
    f1 = (1.0-np.exp(-lambd*z))*theta(tau-z)
    value1 = integrate.trapz(f1,z)
    
    # Note that I_1_adj also allows for theta to be time-dependent,
    # therefore it is not exactly as in the book
    I_1_adj = (i*u-1.0) * value1
    I_2     = tau/(gamma**2.0) *(kappa-gamma*rhoxv*i*u-D1) - \
        2.0/(gamma**2.0)*np.log((1.0-g*np.exp(-D1*tau))/(1.0-g))
    I_3     = 1.0/(2.0*np.power(lambd,3.0))* \
        np.power(i+u,2.0)*(3.0+np.exp(-2.0*lambd*tau) \
        - 4.0*np.exp(-lambd*tau)-2.0*lambd*tau)
    
    meanSqrtV = meanSqrtV_3(kappa,v0,vbar,gamma)
    f2        = meanSqrtV(tau-z)*(1.0-np.exp(-lambd*z))
    value2    = integrate.trapz(f2,z)
    I_4       = -1.0/lambd * (i*u+u**2.0)*value2
    
    return I_1_adj + kappa*vbar*I_2 + 0.5*eta**2.0*I_3+eta*rhoxr*I_4

def C_H1HW(u,tau,lambd):
    i = complex(0.0,1.0)
    C = (i*u - 1.0)/lambd * (1-np.exp(-lambd*tau))
    return C

def D_H1HW(u,tau,kappa,gamma,rhoxv):
    i = complex(0.0,1.0)
    
    D1 = np.sqrt(np.power(kappa-gamma*rhoxv*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = (kappa-gamma*rhoxv*i*u-D1)/(kappa-gamma*rhoxv*i*u+D1)
    C  = (1.0-np.exp(-D1*tau))/(gamma*gamma*(1.0-g*np.exp(-D1*tau)))\
        *(kappa-gamma*rhoxv*i*u-D1)
    return C

def ChFH1HWModel(P0T, lambd, eta, tau, kappa, gamma, vbar, v0, rhoxv, rhoxr):
    # Determine initial interest rate r(0)
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    r0 = f0T(0.00001)
    C = lambda u: C_H1HW(u,tau,lambd)
    D = lambda u: D_H1HW(u,tau,kappa,gamma,rhoxv)
    A = lambda u: A_H1HW(u,tau,P0T,lambd,eta,kappa,gamma,vbar,v0,rhoxv,rhoxr)
    cf = lambda u: np.exp(A(u) + C(u)*r0 + D(u)*v0 )
    return cf

def mainCalculation():
    CP  = OptionType.CALL
    
    # HW model parameter, underlying spot and constant rate 
    lambd = 0.01
    eta   = 0.005
    S0    = 3918 #4031.81
    T     = 1.0       
    r     = -0.0044
    # 10/6/2022	1.005115311	-0.005074467
    # 10/6/2023	1.008864801	-0.004400813
    # 10/7/2024	1.011033471	-0.003644369
   
    # ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-r*T) 

    # Strike range
    K = np.concatenate((np.array([0.001]), S0*np.linspace(.8, 1.2, 5)))

    # Settings for the COS method
    N = 2000
    L = 15  
    
    v0    = 0.02
    kappa = 0.5
    vbar  = 0.05
    gamma = 0.3
    rhoxr = 0.5
    rhoxv =-0.8
    
    
    # The COS method
    cf2 = ChFH1HWModel(P0T, lambd, eta, T, kappa, gamma, vbar, v0, rhoxv, rhoxr)
    valCOS = CallPutOptionPriceCOSMthd_StochIR(cf2, CP, S0, T, K, N, L, P0T(T))
    
    print("Call Value from the COS method:")
    print(valCOS)
    
    CP  = OptionType.PUT
    valCOS_put = CallPutOptionPriceCOSMthd_StochIR(cf2, CP, S0, T, K, N, L, P0T(T))
    print("Put Value from the COS method:")
    print(valCOS_put)

mainCalculation()

# Stock	.STOXX50E		
# Reference Date	10/04/21		
# Spot	4031.81		
			
# Calls	1Y	        2Y	        3Y
# 80%	803.807	    783.6343	766.7052
# 90%	487.1833	503.0326	511.6932
# 100%	233.2686	283.6715	314.8998
# 110%	76.88376	137.0928	179.235
# 120%	17.97704	58.59257	96.16299
			
# 	        1Y	        2Y	        3Y
# Forwards	3897.953	3780.493	3676.727
			
# Puts	1Y	        2Y	        3Y
# 80%	127.7287	223.3451	310.0075
# 90%	216.4284	349.7339	463.0177
# 100%	367.837	    537.3632	674.2464
# 110%	616.7755	797.7749	946.6038
# 120%	963.1922	1126.265	1271.554

# # %% Calib. routine
# # set initial guesses for the parameters
# rt      = 0.01
# kappa   = 1.0
# theta   = 0.0   
# sigma   = 0.05
# params0 = [rt, kappa, theta, sigma]

# # Define optimize\minimize method
# mthd = 'TNC'
# # Methods:
# #   Nelder-Mead (default): Simplex algorithm
# #   Powell: Powell's method of direction set
# #   CG: Non-linear conjugate gradient algorithm
# #   BFGS: Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
# #   L-BFGS-B: Limited-memory BFGS algorithm with box constraints
# #   Newton-CG: Newton's conjugate gradient algorithm 
# #              (Jacobian is required, add "jac=jac")
# #   TNC: Truncated Newton algorithm with bounds and constraints
# #   COBYLA: Constrained optimization by linear approximation
# #   SLSQP: Sequential Least SQuares Programming
# Methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
#            'TNC', 'COBYLA', 'SLSQP']

# # Define the bounds for each parameter
# bnds = ((0.0, 1.0), (0.0, 10), (0.0, 10), (0.0, 10))

# # Define the constraint
# cnst = ({'type': 'ineq', 'fun': lambda x: 2* x[0] * x[1] - x[2] ** 2})

# # Set the options for the solver
# opts = {'disp': True, 'maxiter': 250} # maxfun for some methods

# # optimize the parameters using scipy's 'minimize' function
# for mthd in Methods:
#     result = minimize(lambda x: cir_bond_price_sqr_error(x, T, target_bonds), params0, 
#                     method = mthd, bounds = bnds, constraints = cnst, options = opts)

#     # print results
#     rt, kappa, theta, sigma = result.x
#     print("rt =", rt)
#     print("kappa =", kappa)
#     print("theta =", theta)
#     print("sigma =", sigma)
#     print("2* kappa * theta =", 2* kappa * theta)
#     print("sigma ** 2 =", sigma ** 2)
#     print("Verifies Feller cond. ", 2* kappa * theta >= sigma ** 2)
#     print("Total Sq. error: ", cir_bond_price_sqr_error(result.x, T, target_bonds))
#     print("Methods: ", mthd)

