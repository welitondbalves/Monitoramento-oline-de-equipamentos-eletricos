import numpy as np
from time import sleep
import pandas as pd

#arquivo = pd.read_csv("/home/weliton/Monitoramento_IC/filtra_dados.csv")
#y = arquivo['Configuracao']

def filtra_dados(y):
   quantLinha = len(y)
   seconds = 1           # Constante para definir tempo
   N =  9

   if N <= quantLinha:
      while np.array_equal(y[N],y[N+1]):
         N = N+1
         sleep(seconds)
         if np.array_equal(y[N],y[N+1]):
            mapaAtual = y[N]
         else:
            N = N+1
            
   return mapaAtual

#confAtual = filtra_dados(y)
#print('Configuração atual: ',confAtual)