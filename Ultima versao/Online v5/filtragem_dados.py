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
import csv

def filtragem_dados(data,x):
    mapa_configuracao = 0
    historico = []
    mapa = []
    i = 0

    df = pd.DataFrame(data)
    df.to_csv('dados.csv')
    arquivo = pd.read_csv('dados.csv')
    y = arquivo['0']

    TamanhoY = len(y)-1
    
    print("Tamanho y: ",TamanhoY)

    N = 0  # N linha da tabela de configurações onde inicia a leitura para filtragem
    MAX = TamanhoY # tamanho da lista de configurações para finalizae a leitura da lista

    while(N<MAX):
        mapa_configuracao_gerado = fl.filtra_dados(y,N)        # guarda configuração atual
        if np.array_equal(mapa_configuracao_gerado,mapa_configuracao): # Se não houve mudandaça de configuração mapa e historico atual
            print("N: ",N)
            mapa.insert(N,mapa_configuracao)

        else:
            historico_alteracao = mapa_configuracao
            mapa_configuracao = mapa_configuracao_gerado
            print("N: ",N)
            mapa.insert(N,mapa_configuracao)
            historico.insert(i,mapa_configuracao)
            i = i+1
        
        N = N+1


    historico_lido = []
    df = pd.DataFrame(historico)
    df.to_csv(f'Dados/historico_alteracoes_{x+1}.csv')
    df2 = pd.DataFrame(mapa)
    df2.to_csv(f'Dados/mapa_atual_{x+1}.csv')