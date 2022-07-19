import numpy as np
from time import sleep
import pandas as pd

#arquivo = pd.read_csv("/home/weliton/Monitoramento_IC/filtra_dados.csv")
#y = arquivo['Configuracao']

def filtra_dados(y,n):
   quantLinha = len(y)-1
   #quantLinha = len(y)
   #quantLinha = max
   seconds = 1           # Constante para definir tempo
   N =  n
   mapaAtual = y[N]



   if N < quantLinha:
      #while np.array_equal(y[N],y[N+1]) and ((N+1)<=quantLinha):
      while np.array_equal(y[N],y[N+1]) and ((N+1)<quantLinha):
         N = N+1
         #mapaAtual = y[N]
         sleep(seconds)
         if np.array_equal(y[N],y[N+1]):
            mapaAtual = y[N]
         #else:
            #N = N+1
            #mapaAtual = y[N]
            #break
            
   return mapaAtual
"""
      while np.array_equal(y[N],y[N+1]):
      N = N+1
      #mapaAtual = y[N]
      sleep(seconds)
      if np.array_equal(y[N],y[N+1]):
            mapaAtual = y[N]

   return mapaAtual
"""

#confAtual = filtra_dados(y)
#print('Configuração atual: ',confAtual)