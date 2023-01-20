# Engenharia-medica-aplicada-Unifesp-ICT
Códigos em Phyton utilizados na disciplina de engenharia médica, do curso de Engenharia Biomédica do Instituto de Ciência e Tecnologia - Universidade Federal de São Paulo 

Sumário deste arquivo: 
import numpy as np
import scipy
import scipy.fft as fft
from scipy.signal import welch
import scipy.stats as st
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import colors 
from mpl_toolkits import mplot3d
import time
from itertools import permutations, combinations
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC


'''Sumário

1. Coisas simples
    a)como carregar um arquivo + dar reshape
    b)como printar o tamanho do arquivo
    c)como fazer a matriz transposta
    d)como separar os dados por linha 
    e)como separar os padroes de acordo com as classes
    f)como calcular a média simples ou a média de linha ou coluna HELP
    g)como calcular a variancia simples ou a variancia de linha ou coluna HELP
    h)como plotar um histograma
    i)como calcular sensibilidade, específicidade e probabilidades condicionais
    k)como chamar uma funçao
    l)como construir o espaço de caracteristicas 1D, 2D ou 3D (plotar gráficos)
    m)como utilizar o for (exemplo)
    n)como dar o contrário de "tab"
    o)como criar uma matriz 
    p)como calcular o tempo
    q)como multiplicar matrizes
    r)como contabilizar as amostras (usado após a função classifica perceptron)
    s)como criar um trem de teste
    t)como calcular a covariância
    u)como calcular a matriz de covariância em cada coluna ou linha  

2. Funções
2.1 funções para alterar entradas e/ou espaço de caracteristicas
    a)extraicarac - criação do espaço de caracteristicas 
    b)matrizClasseParaMatrizCaracteristica
    c)concatenarClasses
    d)gerandodadosgaussianos
    e)criarListas
    f)calculaMedeCov
    g)separaClasses

2.2 funções de pré-processamento
    a)encontraOutliers
    b)encontraOutliersComPlot
    c)excluiOutiliers
    d)normaliza
    e)normaliza grupo

2.3 Seleção de caracteristicas e redução de dimensionalidade
    a)TesteEstatisticoParaSelecaoDeClasses
    b)calculaRocAuc - seleção escalar
    c)criterioFDRdeFisher - seleção escalar
    d)indCor - interno
    e)selecaoEscalar - FDR ou ROC- seleção escalar
    f)selecaoEscalarReserva
    g)selecaoVetorialExaustiva
    h)fazPCA
    i)SVD
    j)autoordenar - interno, usado no FDA
    k)scatter2 - iterno, usado no FDA
    l)FDA

2.4 Classificadores
    a)whosMy - interno
    b)classBayesMedCov
    c)valoresPossiveis - interno
    d)classBayesTreino
    e)classificadorDistancia
    f)perceptron
    g)percepoket
    h)matmul - interna, usada no LS
    i)LS
    j)LS2
    k)classificaPorW
    l)plotSeparacaoW
    l.1)classMahaMedCov - interno
    m)preparaSVM
    n)separaSVM

2.5 Avaliação do classificador
    a)analiseEstatistica
    b)Como utilizar o k-fold
    c)Como utilizar o k-fold +FDA

3. Exercícios das aulas
