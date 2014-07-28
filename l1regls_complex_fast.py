# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:38:26 2013

@author: Y1RY
"""
from numpy import array
from cvxopt.base import matrix
from cvxopt import solvers 
from kktchol_fast import mykktchol

def l1regls(A, b, gamma): 
    """ 
        minimize  0.5 * ||A*x-b||_2^2  + gamma*sum_k |x_k| 

    with complex data. 
    """ 

    A=matrix(A)
    b=matrix(b)
    m, n = A.size 

    # Solve as 
    # 
    #     minimize  0.5 * || AA*u - bb ||_2^2 + gamma* sum(t) 
    #     subject   || (u[k], u[k+n]) ||_2 <= t[k], k = 0, ..., n-1. 
    # 
    # with real data and u = ( Re(x), Im(x) ). 

    AA = matrix([ [A.real(), A.imag()], [-A.imag(), A.real()] ]) 
    bb = matrix([ b.real(), b.imag() ]) 
    
    # P = [AA'*AA, 0; 0, 0] 
    P = matrix(0.0, (3*n, 3*n)) 
    P[0:2*n,0:2*n]=AA.T*AA

    # q = [-AA'*bb; gamma*ones] 
    q = matrix([-AA.T * bb, matrix(gamma, (n,1))]) 
    
    # n second order cone constraints || (u[k], u[k+n]) ||_2 <= t[k] 
    I = matrix(0.0, (n,n)) 
    I[::n+1] = -1.0 
    G = matrix(0.0, (3*n, 3*n)) 
    G[1::3, :n] = I 
    G[2::3, n:2*n] = I 
    G[::3, -n:] = I 
    h = matrix(0.0, (3*n, 1)) 

    dims = {'l': 0, 'q': n*[3], 's': []} 
    factor=mykktchol(P)
    sol = solvers.coneqp(P, q, G, h, dims,kktsolver=factor) 
    
    return sol['x'][:n] + 1j*sol['x'][n:2*n] 

if __name__ == '__main__': 
    from cvxopt.base import normal 
    from numpy.random import choice
    m, n = 150, 90
    
    A = normal(m,n) + 1j*normal(m,n) 
    b = normal(n,1) + 1j*normal(n,1) 
    b = array(b)
    b[choice(range(n),3*n/4,False)]=0
    b=matrix(b)
    y=A*b
    gamma = 1.0 
    x = l1regls(A, y, gamma) 