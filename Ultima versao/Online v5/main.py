import os
from math import floor
from time import sleep
import keyboard

# Plotting tools:
import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
import pandas as pd
from nidaqmx.constants import AcquisitionType, Edge
import pickle
from collections import Counter
import filtragem_dados as fd

# Data pre-processing methods:
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

# Data pre-processing methods:
# Conversion of vectors into a binary class matrix:
# from tensorflow.keras.utils import to_categorical
# Indexing of samples for training, validation and test:

#Variáveis globais
dt = 500 #tempo entre observações em milissegundos
n_tempty = 3 #Número de observações consecutivas
fsample = 99960
n_casos = 10 #número de ciclos observados
samples_per_window = 60
data_ = 0

#Definições
MASCARA_VETOR_CARACTERISTICAS = 0
NORM_FUNCTION = 'Minmax'
N_CLASSES = 0
N_AMOSTRAS_POR_CASO = 0
N_AMOSTRAS_POR_CLASSE =0

def setup(path):
    global samples_per_window
    global n_casos
    global fsample
    definitions(path)
    n_casos = int(input('Entre com o número de exemplos por leitura'))
    # Calculate of the number of sample per window
    samples_per_window = int((n_casos/60)*fsample)
    print('A cada ',dt,' ms serão adquiridos ', n_casos, ' ciclos de 60 Hz a uma taxa de amostragem de ', fsample, ' amostras por segundo, resultando em uma massa de dados de ',samples_per_window, ' amostras.')

def daq():
    global data_
    global n_tempty
    data_ = np.array(["voltage", "current"])
    print("data_ shape: ", data_.shape)
    fsample = 60*N_AMOSTRAS_POR_CASO
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0")
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai2")
        task.timing.cfg_samp_clk_timing(fsample, active_edge=Edge.RISING, sample_mode=AcquisitionType.CONTINUOUS,
                                        samps_per_chan=samples_per_window)
        for nw in range(floor(1)):
            data = task.read(number_of_samples_per_channel=samples_per_window)
            data = np.array(data)
            data = data.transpose()
            data_ = np.vstack((data_,data))
    return data

def load_AI(path):
    # load the model from disk
    clf_KNN = pickle.load(open(path + 'clf_KNN.sav', 'rb'))
    clf_SVM = pickle.load(open(path + 'clf_SVM.sav', 'rb'))
    clf_MLP = pickle.load(open(path + 'clf_MLP.sav', 'rb'))
    return [clf_KNN, clf_SVM, clf_MLP]

def predicts(AI, features):
    knn = AI[0]
    svm = AI[1]
    mlp = AI[2]

    pred_knn = knn.predict(features)
    pred_svm = svm.predict(features)
    pred_mlp = mlp.predict(features)
    print([pred_knn, pred_svm, pred_mlp])
    c = Counter(pred_knn)
    c_knn = c.most_common(1)
    c = Counter(pred_svm)
    c_svm = c.most_common(1)
    c = Counter(pred_mlp)
    c_mlp = c.most_common(1)
    return [c_knn, c_svm, c_mlp]

def plot_data(data):
    plt.close("all")  # fecha todos graficos abertos
    fig, axs = plt.subplots(2)
    #fig.suptitle('Vertically stacked subplots')
    #print('data_plot:',data.shape)
    t = np.linspace(0, data.shape[0]/fsample, data.shape[0])
    #print('t', t.shape)
    axs[0].plot(t, data[:,0])
    axs[1].plot(t, data[:,1])
    axs[0].set(ylabel='Voltage [V]')
    axs[1].set(xlabel='Time[s]', ylabel='Current [A]')
    plt.show()

def save_data(data, path, _type='all'):
    if not os.path.exists(path):
        print("Creating", path)
        os.makedirs(path)

    print("Saving the data in ", path, "...")
    # Formato binário
    if _type=='npy':
        np.save(path + "voltage.npy", data[:,0])
        np.save(path + "current.npy", data[:,1])
    # Formato CSV
    if _type == 'csv':
        np.savetxt(path + 'voltage.csv', data[:,0], delimiter=';')
        np.savetxt(path + 'current.csv', data[:,1], delimiter=';')
    # Formato Compactado
    if _type == 'npz':
        np.savez_compressed(path + "data.npz", voltage=data[:,0], current=data[:,1])
    # Todos os 3 formatos
    if _type == 'all':
        np.save(path + "voltage.npy", data[:,0])
        np.save(path + "current.npy", data[:,1])
        np.savetxt(path + 'voltage.csv', data[:,0], delimiter=';')
        np.savetxt(path + 'current.csv', data[:,1], delimiter=';')
        np.savez_compressed(path + "data.npz", voltage=data[:, 0], current=data[:, 1])

def load_data(path, _type='npz'):
    print("Loading data from ", path, "...")
    # Formato binário
    if _type == 'npy':
        voltage = np.load(path + 'voltage.npy')
        current = np.load(path + 'current.npy')
        return ([voltage, current])
    # Formato csv
    if _type == 'csv':
        voltage = np.loadtxt(path + 'voltage.csv', delimiter=';')
        current = np.loadtxt(path + 'current.csv', delimiter=';')
        return ([voltage,current])
    # Formato compactado
    if _type == 'npz':
        dataz = np.load(path + 'data.npz')
        return (dataz)

def _definitions(path):
    # Definição de parâmetros variáveis
    global MASCARA_VETOR_CARACTERISTICAS
    global NORM_FUNCTION
    global N_CLASSES
    global N_AMOSTRAS_POR_CASO
    print('Buscando definições')
    df = pd.read_csv(path + 'DEFINITIONS.csv')
    vetor_características = df['VETOR_CARACTERÍSTICAS']
    MASCARA_VETOR_CARACTERISTICAS = df['MASCARA_VETOR_CARACTERISTICAS'].to_numpy()
    NORM_FUNCTION = 'Minmax' #str(df['NORM_FUNTION'])
    _N_CLASSES = df['N_CLASSES']
    N_CLASSES = int(_N_CLASSES[0])
    _N_AMOSTRAS_POR_CASO = df['RESOLUCAO']
    N_AMOSTRAS_POR_CASO = int(_N_AMOSTRAS_POR_CASO[0])
    print('Máscara de características = ', MASCARA_VETOR_CARACTERISTICAS)
    print('Função de normalização = ', NORM_FUNCTION)
    print('Número de classes = ', N_CLASSES)
    print('Número de amostras por caso = ', N_AMOSTRAS_POR_CASO)

def definitions(path):
    # Definição de parâmetros variáveis
    global MASCARA_VETOR_CARACTERISTICAS
    global NORM_FUNCTION
    global N_CLASSES
    global N_AMOSTRAS_POR_CASO

    MASCARA_VETOR_CARACTERISTICAS = np.asarray([1,1,1,1,1,1])
    NORM_FUNCTION = 'Minmax'
    N_CLASSES = 4
    N_AMOSTRAS_POR_CASO = 1666

def feature_extraction(data):
    # Procedimento para Extração de Características
    global MASCARA_VETOR_CARACTERISTICAS
    global NORM_FUNCTION
    global N_CLASSES
    global N_AMOSTRAS_POR_CASO
    global n_casos

    print('Número de casos =', n_casos)

    vv = data[:, 0]
    ii = data[:, 1]

    tensao_rms = []
    corrente_rms = []
    potencia_ativa = []
    potencia_aparente = []
    potencia_reativa = []
    fator_potencia = []

    for i in range(0, n_casos):
        tensao = vv[i * N_AMOSTRAS_POR_CASO:(i + 1) * N_AMOSTRAS_POR_CASO]
        corrente = ii[i * N_AMOSTRAS_POR_CASO:(i + 1) * N_AMOSTRAS_POR_CASO]

        # Valore médios
        tensao_med = tensao.mean()
        corrente_med = corrente.mean()

        # Quadrado das amostras de corrente e de tensão
        tensao2 = np.power(tensao, 2)
        corrente2 = np.power(corrente, 2)

        # Soma de tensão e de corrente pertencentes a 1 caso
        tensao2_sum = np.sum(tensao2)
        corrente2_sum = np.sum(corrente2)

        # Cálculo da tensão e da corrente rms
        tensao_rms1 = np.sqrt(tensao2_sum / N_AMOSTRAS_POR_CASO)
        corrente_rms1 = np.sqrt(corrente2_sum / N_AMOSTRAS_POR_CASO)
        tensao_rms = np.append(tensao_rms, tensao_rms1)
        corrente_rms = np.append(corrente_rms, corrente_rms1)
        # print('tensao_rms=',tensao_rms1)
        # print('corrente_rms=',corrente_rms1)

        # Cálculo da potência ativa
        potencia_ativa1 = tensao_med * corrente_med
        potencia_ativa = np.append(potencia_ativa, potencia_ativa1)

        # Cálculo do módulo da potência aparente
        potencia_aparente1 = tensao_rms1 * corrente_rms1
        potencia_aparente = np.append(potencia_aparente, potencia_aparente1)
        # print('potencia_aparente=',potencia_aparente1)

        # Cálculo da potência reativa
        potencia_aparente2 = np.power(potencia_aparente1, 2)
        potencia_ativa2 = np.power(potencia_ativa1, 2)
        potencia_reativa1 = np.sqrt(potencia_aparente2 - potencia_ativa2)
        potencia_reativa = np.append(potencia_reativa, potencia_reativa1)
        # print('potencia_reativa=',potencia_reativa1)

        # Cálculo do fator de potencia
        fator_potencia1 = potencia_ativa1 / potencia_aparente1
        fator_potencia = np.append(fator_potencia, fator_potencia1)
        # print('fator de potencia = ', fator_potencia1)

    print('tensao_rms:', tensao_rms.shape)
    print('corrente_rms:', corrente_rms.shape)
    print('potencia_ativa:', potencia_ativa.shape)
    print('potencia_aparente:', potencia_aparente.shape)
    print('potencia_reativa:', potencia_reativa.shape)
    print('fator_potencia:', fator_potencia.shape)

    # Características
    caracteristicas = np.transpose(
        [tensao_rms, corrente_rms, potencia_ativa, potencia_aparente, potencia_reativa, fator_potencia])
    print('caracteristicas:', caracteristicas.shape)
    print(caracteristicas)

    # Seleção de características
    for i in range(caracteristicas.shape[1] - 1, -1, -1):
        if MASCARA_VETOR_CARACTERISTICAS[i] == 0:
            caracteristicas = np.delete(caracteristicas, i, 1)
    print('caracteristicas:', caracteristicas.shape)
    print(caracteristicas)
    return feature_normalize(caracteristicas)

def feature_normalize(caracteristicas):
    # Procedimento para normalização

    if NORM_FUNCTION == 'Minmax':
        normalizer = MinMaxScaler()
    elif NORM_FUNCTION == 'StandardScaler':
        normalizer = StandardScaler()
    elif NORM_FUNCTION == 'PowerTransformer':
        normalizer = PowerTransformer(method='yeo-johnson', standardize=True)

    # caracteristicas = np.asarray([[1, 2, 3],[20,1,5]])
    # print(caracteristicas.shape)

    caracteristicas_normalizadas = normalizer.fit_transform(caracteristicas)
    print('caracteristicas_normalizadas:', caracteristicas_normalizadas.shape)
    print('caracteristicas_normalizadas=', caracteristicas_normalizadas)
    return caracteristicas_normalizadas

def data_preprocessing(data):
    print('Data_processing')
    definitions(data)
    feature = feature_extraction(data)
    return feature

def main():
    hitorico_main = []
    i = 0
    path = r'C:\Users\Aluno 3\Weliton\Online v5\IAs/'
    
    while keyboard.is_pressed('esc') == False:
        setup(path)
        data = daq()
        print('data.shape:',data.shape)
        print('data:',data)
        six_cycles = int(fsample/10)
        plot_data(data[0:six_cycles,:])
        print("Iniciando pré-processamento...")
        features = data_preprocessing(data)
        print('Features: ',features.shape)
        print(features)
        AI = load_AI(path)
        print('IAs carregadas.')
        pred = predicts(AI, features)
        fd.filtragem_dados(pred,i)
        print('Predições:',pred)
        i = i+1
        sleep(1)
    else:
        print('Saindo do loop principal porque pressionou esc')

if __name__ == '__main__':
    main()