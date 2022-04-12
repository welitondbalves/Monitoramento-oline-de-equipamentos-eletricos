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
#seconds = 1           # Constante para definir tempo

#x = pd.read_csv("/home/weliton/Monitoramento_IC/terra.csv")
#confAtual = x['Configuracao']

#arquivo = pd.read_csv("/home/weliton/Monitoramento_IC/filtra_dados.csv")
#y = arquivo['Configuracao']

#while np.array_equal(y,y):
#    print("Inguais")
#    break

#print("Diferentes")
#b = 1
while(False):
    if keyboard.is_pressed('esc'):
        arquivo = pd.read_csv("/home/weliton/Monitoramento_IC/filtra_dados.csv")# salva informações em csv
        y = arquivo['Configuracao']
        """
        def filtra_dados(y):
            #frequencias = Counter(y).most_common()           # Retorna a configuração com maior numero de leitura
            #mais_frequencia = frequencias[0]
            while ( b == 1):      # ver se é assim mesmo
                if np.array_equal(y,confAtual):
                    print("Iguais")

                else:
                    sleep(seconds)
                    confAtual = y
                    print("Diferentes")
                
                b = 2

            return confAtual
        """
"""
        mapa_configuracao_gerado = filtra_dados(y)            # guarda configuração atual
        if(mapa_configuracao_gerado == mapa_configuracao):   # Se não houve mudandaça de configuração mapa e historico atual

            print(mapa_configuracao)                         # mostra mapa e historico atual
            print(arquivo_dados)
        
        else:                                                # Se houve mudança de configuração
            historico_alteracao[] = mapa_configuracao        # atualiza historico de atualização
            mapa_configuracao = mapa_configuracao_gerado     # atualiza mapa
            arquivo_dados = historico_alteracao[]            # atualiza dados

            print(mapa_configuracao)                         # mostra mapa e historico atual
            print(arquivo_dados)
 

    sleep(seconds)                                                # aguarda 1 segundo
"""