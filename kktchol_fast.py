# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:29:04 2014

@author: Rafael
"""
from cvxopt import matrix,blas,lapack
from cvxopt.misc import scale
import WGGW_C
from numpy import zeros

Gs=matrix(0.0,(1,0))
K=matrix(0.0,(1,0))
vs=zeros((0,0))
betas=zeros((0,))

def mykktchol(H):
    """
    Solution of KKT equations by reduction to a 2 x 2 system, a QR 
    factorization to eliminate the equality constraints, and a dense 
    Cholesky factorization of order n-p. 
    
    Returns a function that (1) computes the Cholesky factorization 
    
        H + GG^T * W^{-1} * W^{-T} * GG = L * L^T, 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H    GG'    ]   [ ux ] = [ bx ]
        [ GG   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """
    global vs,betas,K,Gs

    n=H.size[0]
    Gs=matrix(0.0,(n,n))
    K=matrix(0.0,(n,n))
    vs=zeros((n,1))
    betas=zeros((n/3,))
    
    def factor(W):
        global vs,K,Gs
        # Gs = W^{-T} G
        # K = Gs^T Gs    
        for k in range(n/3):
            betas[k]=W['beta'][k]
            vs[3*k]=W['v'][k][0]
            vs[3*k+1]=W['v'][k][1]
            vs[3*k+2]=W['v'][k][2]
        
        #K= P + G^T W^{-1} W^{-T} G
        #K[:,:] += H

        # Cholesky factorization of K.
        #lapack.potrf(K, n = n, offsetA = 0)

        WGGW_C.makeGsAndK(vs,betas,Gs,K,H,n)
        
        def solve(x, y, z):

            # Solve
            #
            #     [ P           GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ W^{-T}*GG   -I         ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.
            #
            # z=W^-T z
            scale(z, W, trans = 'T', inverse = 'I')

            # x=x+Gs^T z
#            blas.gemv(Gs, z, x, beta = 1.0, trans = 'T', m = n)          
            WGGW_C.Gs_mv(Gs, z, x, 1,1, n)
            
            #solve Kx
            lapack.potrs(K, x, n = n, offsetA = 0, offsetB = 0)

            # z=z-Gs^T x
            #blas.gemv(Gs, x, z, alpha = 1.0, beta = -1.0, m = n)
            WGGW_C.Gs_mv(Gs, x, z, -1,0, n)
            

        return solve

    return factor