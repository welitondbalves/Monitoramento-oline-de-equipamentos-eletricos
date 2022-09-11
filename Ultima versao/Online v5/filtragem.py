import numpy as np
from time import sleep
import pandas as pd


def filtra_dados(y,n):
   quantLinha = len(y)-1
   
   seconds = 1           # Constante para definir tempo
   N =  n
   mapaAtual = y[N]



   if N < quantLinha:
      
      while np.array_equal(y[N],y[N+1]) and ((N+1)<quantLinha):
         N = N+1
         
         sleep(seconds)
         if np.array_equal(y[N],y[N+1]):
            mapaAtual = y[N]
            
   return mapaAtual