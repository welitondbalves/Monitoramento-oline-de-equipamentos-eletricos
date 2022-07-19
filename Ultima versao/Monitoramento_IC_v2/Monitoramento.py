from time import sleep
from tkinter import Y
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
import keyboard
import numpy as np
import filtragem as fl

mapa_configuracao = 0
#mapa_configuracao_gerado = 0
#seconds = 1           # Constante para definir tempo

x = pd.read_csv("/home/weliton/Monitoramento_IC/terra.csv")
confAtual = x['Configuracao']

arquivo = pd.read_csv("/home/weliton/Monitoramento_IC/filtra_dados.csv")
y = arquivo['Configuracao']


# uma variavel x para receber terra.csv

#while np.array_equal(y,y):
#    print("Inguais")
#    break

#print("Diferentes")
#b = 1
N = 0  # N linha da tabela de configurações onde inicia a leitura para filtragem
 
while(False):
    if keyboard.is_pressed('esc'):
        #arquivo = pd.read_csv("/home/weliton/Monitoramento_IC/filtra_dados.csv")# salva informações em csv
        #y = arquivo['Configuracao']
        
        #mapa_configuracao_gerado = filtra_dados(y)            # guarda configuração atual
        mapa_configuracao_gerado = fl.filtra_dados(y,N)        # guarda configuração atual
           
        if np.array_equal(mapa_configuracao_gerado,mapa_configuracao): # Se não houve mudandaça de configuração mapa e historico atual

            # Y vai receber a configuração filtrada e inserir na linha atual do arquivo csv
            print(mapa_configuracao)                         # mostra mapa e historico atual
            #print(confAtual[N])
        
        else:                                                # Se houve mudança de configuração
            historico_alteracao = mapa_configuracao        # atualiza historico de atualização
            mapa_configuracao = mapa_configuracao_gerado     # atualiza mapa
            arquivo_dados = historico_alteracao            # atualiza dados
            confAtual[N] = historico_alteracao 
            # quardar historico_alteracao na variavel x
            # x -> dataframe
            # dataframe para csv
            
            # Y vai receber a configuração filtrada e inserir na linha atual do arquivo csv
            print(mapa_configuracao)                         # mostra mapa e historico atual
            #print(confAtual[N])
 

    N = N+1
    sleep(seconds)                                                # aguarda 1 segundo
