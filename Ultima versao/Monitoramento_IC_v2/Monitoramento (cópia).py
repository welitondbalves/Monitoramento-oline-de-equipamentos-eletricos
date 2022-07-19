from time import sleep
from tkinter import Y
from tkinter.tix import MAX
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
import keyboard
import numpy as np
import filtragem as fl

mapa_configuracao = 0
historico = []
i = 0
#mapa_configuracao_gerado = 0
#seconds = 1           # Constante para definir tempo

x = pd.read_csv("/home/weliton/Monitoramento_IC_v2/terra.csv")
confAtual = x['Configuracao']

arquivo = pd.read_csv("/home/weliton/Monitoramento_IC_v2/filtra_dados.csv")
y = arquivo['Configuracao']
#arquivo = pd.read_csv("/home/weliton/Monitoramento_IC_v2/filtra_dados_2.csv")
#y = arquivo['Configuracao']

#TamanhoY = len(y)-9 # variavel com o tamanho da coluna y com as configurações lidas
TamanhoY = len(y)-1
#TamanhoY = len(y)
print("Tamanho y: ",TamanhoY)

# uma variavel x para receber terra.csv

#while np.array_equal(y,y):
#    print("Inguais")
#    break

#print("Diferentes")
#b = 1
N = 0  # N linha da tabela de configurações onde inicia a leitura para filtragem
MAX = TamanhoY # tamanho da lista de configurações para finalizae a leitura da lista 
#MAX = 10
while(N<MAX):
    mapa_configuracao_gerado = fl.filtra_dados(y,N)        # guarda configuração atual
    if np.array_equal(mapa_configuracao_gerado,mapa_configuracao): # Se não houve mudandaça de configuração mapa e historico atual
        print("if: ")
        print("Configuracao atual",mapa_configuracao)
        print("N: ",N)
        print(confAtual[N])
        #historico.insert(N,mapa_configuracao)

    else:
        historico_alteracao = mapa_configuracao
        mapa_configuracao = mapa_configuracao_gerado
        confAtual[N] = historico_alteracao
        print("else: ")
        print("Configuracao atual",mapa_configuracao)
        print("N: ",N)
        print(confAtual[N])
        historico.insert(i,mapa_configuracao)
        i = i+1

    N = N+1

df = pd.DataFrame(historico)
df.to_csv("/home/weliton/Monitoramento_IC_v2/alteracoes_configuracao.csv")
