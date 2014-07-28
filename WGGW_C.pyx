# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:14:52 2014

@author: Rafael
"""
import numpy as np
import scipy.linalg.lapack as lpk
from cvxopt import matrix
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL

ctypedef int dpotrf_t(char *uplo, int *n, double *a, int *lda, int *info)
    
cdef dpotrf_t *dpotrf = <dpotrf_t*>f2py_pointer(lpk.dpotrf._cpointer) 

def makeGsAndK(np.ndarray[DTYPE_t,ndim=2]vs,np.ndarray[DTYPE_t,ndim=1]betas,double [:,:] WG, double [:,:] P, double [:,:] H,int n):
    makeGsAndK_impl(vs, betas, WG, P, H, n)   
    
@cython.boundscheck(False) 
cdef void makeGsAndK_impl(DTYPE_t [:,:] vs,DTYPE_t [:] betas,double [:,:] WG, double [:,:] P, double [:,:] H, int n):
    cdef Py_ssize_t rg=n/3
    cdef Py_ssize_t i
    cdef Py_ssize_t j1
    cdef Py_ssize_t j2
    cdef Py_ssize_t k
    cdef DTYPE_t a,b,c,beta,wggw

    with nogil:
        for k in range(rg):
            beta=1.0/betas[k]
            vs[3*k,0]=-vs[3*k,0]
            a=vs[3*k,0]
            b=vs[3*k+1,0]
            c=vs[3*k+2,0]
            WG[3*k,k+2*rg]=-beta*(2*a*a-1)
            WG[3*k+1,k]=-beta*(2*b*b+1)
            WG[3*k+2,k+rg]=-beta*(2*c*c+1)
            WG[3*k+1,k+2*rg]=WG[3*k,k]=-2*beta*a*b
            WG[3*k+2,k+2*rg]=WG[3*k,k+rg]=-2*beta*a*c
            WG[3*k+2,k]=WG[3*k+1,k+rg]=-2*beta*b*c

    P[...]=H
    
    with nogil:            
        for k in range(rg):
            for i in range(3):
                for j1 in range(3):
                    P[j1*rg+k,j1*rg+k]+=WG[3*k+i,j1*rg+k]*WG[3*k+i,j1*rg+k]
                    for j2 in range(j1):
                        wggw=WG[3*k+i,j2*rg+k]*WG[3*k+i,j1*rg+k]
                        P[j1*rg+k,j2*rg+k]+=wggw
                        P[j2*rg+k,j1*rg+k]+=wggw

    cdef int info=0
    dpotrf('L',&n,&P[0,0],&n,&info)

def Gs_mv(DTYPE_t [:,:] Gs, DTYPE_t [:,:] z, DTYPE_t [:,:] x, int sign,int trans, int n):
    Gs_mv_impl(Gs, z, x, sign, trans, n)

@cython.boundscheck(False)     
cdef Gs_mv_impl(DTYPE_t [:,:] Gs, DTYPE_t [:,:] z, DTYPE_t [:,:] x, int sign,int trans, int n):
    cdef Py_ssize_t rg=n/3
    cdef Py_ssize_t k=0
    
    with nogil:
        for k in range(rg):
            if trans!=0:
                if (sign>=0):
                    x[k+2*rg,0]+=Gs[3*k  , k+2*rg]*z[3*k   ,0]
                    x[k     ,0]+=Gs[3*k+1, k     ]*z[3*k+1 ,0]
                    x[k+rg  ,0]+=Gs[3*k+2, k+rg  ]*z[3*k+2 ,0]
                    x[k+2*rg,0]+=Gs[3*k+1, k+2*rg]*z[3*k+1 ,0]
                    x[k     ,0]+=Gs[3*k  , k     ]*z[3*k   ,0]
                    x[k+2*rg,0]+=Gs[3*k+2, k+2*rg]*z[3*k+2 ,0]
                    x[k+rg  ,0]+=Gs[3*k  , k+rg  ]*z[3*k   ,0]
                    x[k     ,0]+=Gs[3*k+2, k     ]*z[3*k+2 ,0]
                    x[k+rg  ,0]+=Gs[3*k+1, k+rg  ]*z[3*k+1 ,0]
                else:
                    x[k+2*rg,0]-=Gs[3*k  , k+2*rg]*z[3*k   ,0]
                    x[k     ,0]-=Gs[3*k+1, k     ]*z[3*k+1 ,0]
                    x[k+rg  ,0]-=Gs[3*k+2, k+rg  ]*z[3*k+2 ,0]
                    x[k+2*rg,0]-=Gs[3*k+1, k+2*rg]*z[3*k+1 ,0]
                    x[k     ,0]-=Gs[3*k  , k     ]*z[3*k   ,0]
                    x[k+2*rg,0]-=Gs[3*k+2, k+2*rg]*z[3*k+2 ,0]
                    x[k+rg  ,0]-=Gs[3*k  , k+rg  ]*z[3*k   ,0]
                    x[k     ,0]-=Gs[3*k+2, k     ]*z[3*k+2 ,0]
                    x[k+rg  ,0]-=Gs[3*k+1, k+rg  ]*z[3*k+1 ,0]
            else:
                if (sign>=0):
                    x[3*k  ,0]+=Gs[3*k  , k+2*rg]*z[k+2*rg ,0]
                    x[3*k+1,0]+=Gs[3*k+1, k     ]*z[k      ,0]
                    x[3*k+2,0]+=Gs[3*k+2, k+rg  ]*z[k+  rg ,0]
                    x[3*k+1,0]+=Gs[3*k+1, k+2*rg]*z[k+2*rg ,0]
                    x[3*k  ,0]+=Gs[3*k  , k     ]*z[k      ,0]
                    x[3*k+2,0]+=Gs[3*k+2, k+2*rg]*z[k+2*rg ,0]
                    x[3*k  ,0]+=Gs[3*k  , k+rg  ]*z[k+rg   ,0]
                    x[3*k+2,0]+=Gs[3*k+2, k     ]*z[k      ,0]
                    x[3*k+1,0]+=Gs[3*k+1, k+rg  ]*z[k+rg   ,0]
                else:
                    x[3*k  ,0]-=Gs[3*k  , k+2*rg]*z[k+2*rg ,0]
                    x[3*k+1,0]-=Gs[3*k+1, k     ]*z[k      ,0]
                    x[3*k+2,0]-=Gs[3*k+2, k+rg  ]*z[k+  rg ,0]
                    x[3*k+1,0]-=Gs[3*k+1, k+2*rg]*z[k+2*rg ,0]
                    x[3*k  ,0]-=Gs[3*k  , k     ]*z[k      ,0]
                    x[3*k+2,0]-=Gs[3*k+2, k+2*rg]*z[k+2*rg ,0]
                    x[3*k  ,0]-=Gs[3*k  , k+rg  ]*z[k+rg   ,0]
                    x[3*k+2,0]-=Gs[3*k+2, k     ]*z[k      ,0]
                    x[3*k+1,0]-=Gs[3*k+1, k+rg  ]*z[k+rg   ,0]

        if sign<0:
            for k in range(n):
                x[k,0]=-x[k,0]

