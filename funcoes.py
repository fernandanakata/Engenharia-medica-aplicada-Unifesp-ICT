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

4. Resumo da teoria


'''
'''Coisas simples

--------------------------------------------------------------
a)como carregar um arquivo + dar reshape
arquivo = scipy.io.loadmat('endereço do arquivo') #Exemplo: 'Aula 10\Dados1.mat'
dados = arquivo['dados']
classes = arquivo['classes']

*caso o arquivo seja algo do tipo (1,N) ou (N,1) (onde N>1), dar reshape(-1) para obter algo de apenas 1 dimensão. 
Isso acontece principalmente em dados originários do matlab, que costuam ter duas dimensões. 
É especialmente útil nas variáveis de classe. 
Nesse caso:
classes = arquivo['classes'].reshape(-1)
--------------------------------------------------------------
b) como printar o tamanho do arquivo
print (np.shape(arquivo))
--------------------------------------------------------------
c)como fazer a matriz transposta
-> a trasposta troca linhas e colunas de lugar. É util para transformar coisas no formato (caracteristicasxpadrões) em (padrõesxcaracteristiscas)
np.transpose(matriz)
--------------------------------------------------------------
d)como separar os dados por linha 
é bem simples:
linha1 = dados[0,:]
linha2 = dados[1,:]
linha3 = dados[2,:]
linha4 = dados[3,:]
*nota: o phyton começa a contar do 0, em vez do 1. 
*nota: a ordem é [linhas,colunas]
*nota: ":" significa selecionar todos os elementos daquela linha ou coluna
--------------------------------------------------------------
e)como separar os padroes de acordo com as classes 
usar a função "separaclasses" ou fazer que: 
sendo "padrões" dados do tipo (1600x4) (padrõesxcaracteristicas) e classes do tipo (1600x1), com classes iguais a -1 e 1:
classe1=padroes[classes==-1,:]
classe2=padroes[classes==2,:]
--------------------------------------------------------------
f)como calcular a média simples ou a média em cada linha ou coluna
Exemplo para média simples: 
notas = [7.5, 8.3, 9.0, 6.5]
media = np.mean(notas)
print(media)
--------------
Exemplo para média em cada coluna ou linha:
notas = [[7.5, 8.3, 9.0, 6.5],[1,2,3,4]]

mediaLinhas = np.mean(notas,axis=1) #axis 0 para coluna, axis 1 para linha
mediaColunas = np.mean(notas,axis=0)
print (mediaLinhas) #vetor com a quantidade de linhas (nesse caso, 2 casas) com a média de cada linha
print (mediaColunas) #vetor com a média de cada coluna (nesse caso, 4 colunas)

#para pegar a média da 1º linha, por exemplo:
print (mediaLinhas[0]) 
--------------------------------------------------------------
g)como calcular a variancia simples ou a vaciancia em cada linha ou coluna
Exemplo para variancia simples:   
notas = [7.5, 8.3, 9.0, 6.5]
var = np.var(notas) 
print(var)
--------------
Exemplo para variancia em cada coluna ou linha:
notas = [[7.5, 8.3, 9.0, 6.5],[1,2,3,4]]

varLinhas = np.var(notas,axis=1) #axis 0 para coluna, axis 1 para linha
varColunas = np.var(notas,axis=0)
print (varLinhas) #vetor com a quantidade de linhas (nesse caso, 2 casas) com a var de cada linha
print (varColunas) #vetor com a var de cada coluna (nesse caso, 4 colunas)

#para pegar a variância da 1º linha, por exemplo:
print (varLinhas[0]) 
--------------------------------------------------------------
h)como plotar um histograma
plt.title('Título do histograma')
plt.hist(classe[:,0],rwidth=10, color='pink') #queremos plocar a coluna 1 da matriz classe(padrõesxcaracteristicas)
plt.show()   
-------------------------------------------------------------- 
i)como calcular sensibilidade, específicidade e probabilidades condicionais
vn = deu 0 e é 0
fp = deu 1 e é 0
vp = deu 1 e é 1
fn = deu 0 e é 1
--------------
sensibilidade: vp/(vp+fn)
Especificidade:vn/(vn+fp)
--------------
Probabilidades condicionais. Exemplo:
Qual a probabilidade de um paciente estar doente sendo que o seu exame PSA deu positivo 
e a prevalência da doença para o seu grupo é de 4,2%?

Raciocínio:
P(T=1|D=1)=25,57% (sensibilidade)
P(T=0|D=1)=74,43%
*
P(T=0|D=0)=94,91% (especificidade)
P(T=1|D=0)=5,09%
*
P(D=1)=4,2% (prevalência) (ou 0.042)
P(D=0)=95,8%

Queremos P(D=1|T=1)
P(D=1|T=1)=(P(T=1|D=1)*P(D=1))/P(T=1)
Onde P(T=1) é P(T=1,D=1) + P(T=1,D=0)
Para encontrar as probabilidades conjuntas, usamos que: (P(x|y)=(P(x,y))/P(y))
Portanto, P(T=1) = P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)
Portanto, P(D=1|T=1) = (P(T=1|D=1)*P(D=1))/P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)

Portanto, P(D=1|T=1) = P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)
Assim, no código:

probabilidade1 = ((sensibilidade1*0.042))/((sensibilidade1*(4.2/100))+(5.09/100)*95.8/100)
print ("a probabilidade é", probabilidade1*100,"%")
-------------------------------------------------------------- 
k)como chamar uma função
import funcoes #nome da função. Precisa estar na mesma pasta do arquivo
saída1,saída2, etc = funcoes.nomedafuncao(entrada1, entrada2, entrada3)
Caso a função não receba nada, é só chama-la. Exemplo: funcoesprovav4.plotSeparacaoW(dadosT, w2)
-------------------------------------------------------------- 
l)como construir o espaço de caracteristicas 1D, 2D ou 3D (plotar gráficos)
->1D
Exemplo:
plt.figure()
plt.title('título')
plt.xlabel('carac1')
plt.ylabel('carac2')
plt.plot (classe1FDA[:,0],"bo")
plt.plot (classe2FDA[:,0],"ro")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()
--------------
->2D tendo duas matrizes do tipo (padraoxcaracteristicas), uma para cada classe (classe1 e classe2)
plt.figure()
plt.title('Espaço de caracteristicas')
plt.xlabel('Mobilidade')
plt.ylabel('Complexidade')
plt.plot (classe1[:,2],classe1[:,3],"bo") #queremos plotar as caracteristicas que estão nas colunas 2 e 3
plt.plot (classe5[:,2],classe5[:,3],"ro")
plt.plot (classe6[:,2],classe6[:,3],"go")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()
--------------
-> 3D
fig = plt.figure(figsize=(5,5))
plt.title('Espaço de caracteristicas')
ax = fig.add_subplot(111, projection='3d')

x1 = classe1[:,2] #queremos plotar as caracteristicas 2,4 e 8 da classe 1
y1 = classe1[:,4] #queremos as colunas/caracteristicas 2,4 e 8 da classe 1 
z1 = classe1[:,8] #como são 3 caracteristicas, são 3D

x2 = classe5[:,2] #queremos plotar as caracteristicas 2,4 e 8 da classe 5
y2 = classe5[:,4]
z2 = classe5[:,8]

x3 = classe6[:,2] #queremos plotar as caracteristicas 2,4 e 8 da classe 6
y3 = classe6[:,4]
z3 = classe6[:,8]

ax.scatter(x1, y1, z1,color='pink',alpha=0.5)
ax.scatter(x2, y2, z2,color='blue',alpha=0.5)
ax.scatter(x3, y3, z3,color='green',alpha=0.5)
plt.show()
-------------------------------------------------------------- 
m)como utilizar o for (exemplo)
Exemplo da aula 6, no cálculo da sensibilidade. 
l,c = np.shape(dados) #l:linhas, c:colunas
vp1 = (0)
fn1 = (0)

for i in range (l):
    if (coluna1[i]==coluna3[i]==[1]):
        vp1 = (vp1+1)
    elif (coluna1[i]==0 and coluna3[i]==1):
        fn1 = (fn1+1)
        
sensibilidade1 = (vp1/(vp1+fn1)) #usando a saída do for
-------------------------------------------------------------- 
n)para dar o contrário de "tab"
ctrl+"["
-------------------------------------------------------------- 
o)como criar uma matriz 

essa é uma matriz (3x3)
matriz = np.array[[a,b,c],[d,e,f],[g,h,i]] 
#cada [] é uma linha da matriz

essa é uma matriz 3x4
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
-------------------------------------------------------------- 
p)como calcular o tempo 
t1=time.time()
função que queremos saber o tempo de processamento
t2=time.time()
print (t2-t1) 
--------------------------------------------------------------
q)como multiplicar matrizes? 
- util para transformar os dados de teste pós FDA
#converte os dados de teste para o novo sistema de eixos. 
    
.dot é uma operação matricial. Basicamente ela multiplica matrizes. 
dadosTesteTransformados = dadoTeste.dot(autovetores)
onde autovetores é o vetor pós FDA.  
+ na função FDA
--------------------------------------------------------------
r)como contabilizar as amostras (em geral usado, após a função classifica perceptron)
resultadosTeste = funcoesprovav4.classificaPorW(dTeste,w2)
print ("os dados gerados pelo trem de teste pertencem à classe 1?",resultadosTeste)

#Contabilizando
tamanho = np.shape(resultadosTeste)
grupo1 = np.sum(resultadosTeste) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("\n")
print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")
--------------------------------------------------------------
s)como criar um trem de teste (dados aleatórios para testar o classificador)
    from sklearn.model_selection import train_test_split
    gabaritoTreino, gabaritoTeste, grupoTreino, grupoTeste = train_test_split(classes, padroes, train_size = 0.8, random_state = 46)
    #a coisa importante aqui é o grupoTeste
    dTeste = grupoTeste
--------------------------------------------------------------
t)como calcular a covariância
cov=np.cov(dado)
--------------------------------------------------------------  
u)Como calcular a matriz de covariância em cada coluna ou linha   
--------------------------------------------------------------    
'''
################# FUNCOES PARA ALTERAR ENTRADA OU ESPAÇO DE CARACTERISTICAS


def extraicarac(sin,freq,bandas,method='fft',nwelch=5):
  '''
   Definição: Extrai características estatísticas e espectrais de um conjunto de 
   sinais de mesma dimensão temporal
   
   Entradas:
    ----------
    - sin : sinal de onde queremos extrair as caracteristicas(padroesxtempo)
    - freq : frequencia de amostragem dos sinais (Hz). Lembrando que F=1/período
    - bandas : dicionario com a informação das bandas de frequencia a serem extraídas. 
        -> é preciso definir um vetor de bandas antes de chamar a função. 
        -> exemplo de vetor de bandas:
        ->bandas={'delta 1':[0.5,2.5],'delta 2':[2.5,4],'teta 1':[4,6],'teta 2':[6,8], 'alfa':[8,12],'beta':[12,20],'gama':[20,45]}
        -> o formato genérico do vetor de bandas é:
        bandas:{'nome da banda (string)':[freqinicial, freqfinal]}
    - method : 'fft' or 'welch' (se welch, "nwelch" é o numero de trechos no qual o sinal é dividido)
    - nwelch : TYPE, optional. Fica setado em 5. 
    
    Saídas:
    -------
    - car:array de trechos x características
    - nomesc:vetor com os nomes das caracteristicas correspondentes ao array
  '''
  import numpy as np
  import scipy.fft as fft
  (S,X)=np.shape(sin) #S = numero de sinais sinais; X = tamanho dos sinais no tempo
  nc=8+len(bandas) #numero de caracteristicas que serao extraidas
  car=np.zeros((S,nc)) #matriz das caracteristicas
  nomesc=[None]*nc
  
  for s in range(S):

    #média
     car[s,0]=np.mean(sin[s,:])
     nomesc[0]='media'

    #variancia
     var0=np.var(sin[s,:],ddof=1)
     car[s,1]=var0
     nomesc[1]='variancia'

    #mobilidade
     x1=np.diff(sin[s,:])
     var1=np.var(x1,ddof=1)
     mob=var1/var0
     car[s,2]=mob
     nomesc[2]='mobilidade'
          
    #complexidade estatística
     x2=np.diff(x1)
     var2=np.var(x2,ddof=1)
     ce=(var2/var1-var1/var0)**(1/2)
     car[s,3]=ce
     nomesc[3]='complexidade'

    ##calculando o espectro:
     if method=='fft':
       yf = np.abs(fft.rfft(sin[s,:]-car[s,0]))**2 
       yf=yf/X
       yf=yf[0:int(np.floor(X/2)+1)]
       xf = np.linspace(0.0, 1.0/(2.0/freq), len(yf))  
     elif method=='welch':
       xf,yf = welch(sin[s,:]-car[s,0],freq,nperseg=X//nwelch)   
     Yf=yf/np.sum(yf) 

    #frequência central do espectro
     car[s,4]=np.sum(xf*Yf)
     nomesc[4]='f-central'

    #potencia na frequencia central
     ifc=np.abs(xf-car[s,4])==np.min(np.abs(xf-car[s,4]))
     car[s,5]=yf[ifc]
     nomesc[5]='P na fc'

    #largura de banda do espectro
     car[s,6]=np.sqrt(np.sum(((xf-car[s,4])**2)*Yf))
     nomesc[6]='l-banda'
    
    #frequência de margem do espectro
     sw=np.cumsum(Yf)
     f=np.max(np.where(sw<=0.9)[0])
     car[s,7]=xf[f]
     nomesc[7]='f-margem'

    #potências espectrais normalizadas nas seguintes bandas: 
    #delta 1 (0.5 a 2.5Hz)
     for ib, b in enumerate(bandas):
        car[s,8+ib]=sum(Yf[((xf>=bandas[b][0]) & (xf<=bandas[b][1]))])
        nomesc[8+ib]='%'+b

  return (car,nomesc)

def matrizClasseParaMatrizCaracteristica (*args):
    '''
    a função recebe infinitas matrizes (NxL) e converte em matrizes (NxC) ou vice versa
    obs: todas as matrizes precisam ter o mesmo formato e numero de padroes
    
    Entradas:várias matrizes do tipo padrõesxclasses, do mesmo tamanho e diversas classes diferentes
    Saídas:várias matrizes do tipo padrõesxcaracteristicas, uma para cada caracteristica. 
    
    ------
    Exemplo de como funciona:
        
    Tendo: (padrãoxclasse)
    caracteristica 1	        caracteristica 2
    classe 1  |	classe 2       classe 1	|  classe 2
        a     |      d            g     |     j
        b	  |      e            h     |     k
        c	  |      f            i     |     l

    Transforma em: (padrãoxcaracteristica)
       
    classe1	                                             classe 2
    caracteristica 1  |	cacarcteristica 2       caracteristica 1	|  caracteristica 2
                a     |      d                                g     |     j
                 b	  |      e                                h     |     k
                 c	  |      f                                i     |     l

    --------
    obs -> depois de usar, precisamos separar as classes. 
    Exemplo de aplicação:
        EXEMPLO DE COMO APLICAR, DA AULA 10:
        matrizes = funcoes_AT10.matrizClasseParaMatrizCaracteristica(med,ske)
        classe1 = matrizes[0]
        classe2 = matrizes[1]
    
    '''
    matrizes=[]
    n,l=np.shape(args[0])
    c=np.shape(args)[0]
    for i in range(l):
        matrizCarac=np.zeros((n,c))
        for j in range(c):
            matriz=args[j]
            N,L=np.shape(matriz)
            if(n==N):
                matrizCarac[:,j]=matriz[:,i]
            else:
                print("as matrizes não tem o mesmo numero de amostras")
                return
        matrizes.append(matrizCarac)
    return matrizes

def concatenarClasses(*args):
    #a função recebe infinitas matrizes (padrõesxcaracteristicas) de classes distintas mas mesmas colunas (quantidades de caracteristicas)
    #retorna uma unica matriz com todas as classes unificadas e um vetor gabarito que permite a separação delas 
    
    #Para 'desconcatenar' (exemplo)
        #classe1 = classesConcatenadasNormalizadas[gabarito==0,:] #classesConcatenadasNormalizadas é classesConcatenadas após passar pela função normalização. 
        #classe2 = classesConcatenadasNormalizadas[gabarito==1,:]
    
    c=np.shape(args)[0]
    classesConcatenadas=0
    gabarito=0
    for i in range(c):
        n,l=np.shape(args[i])
        mat=args[i]
        gab=i*np.ones(n)
        if( i==0 ):
            classesConcatenadas=mat
            gabarito=gab
        else:
            classesConcatenadas=np.concatenate((classesConcatenadas,mat),axis=0)
            gabarito=np.concatenate((gabarito,gab))
    
    return classesConcatenadas,gabarito

def gerandodadosgaussianos(medias,covariancias,N,priors=0,plotar=True, seed=0,angulo=[0,0]):
    '''
    Gera um conjunto de dados simulados que seguem uma distribuição gaussiana. 
    As classes possuem médias e covariâncias distintas. 
    
    Entradas:
        medias = uma lista com as médias de cada classe. 
        usar a função criarLista 
        -> exemplo (meu)
        media1 = (-2,6,6)
        media2 = (2,6,6)
        listaMedias = funcoes.criarListas(media1,media2)
        -> exemplo (do professor) 
        medias = np.array ([[-2,6,6],[2,6,6]]) -> cada [] é uma linha
        
        covariâncias = uma lista com as covariâncias de cada classe
        -> exemplo (meu)
        cov1 = [[0.3, 1,1],[1,9,1], [1,1,9]] #criando a matriz. cada [] é uma linha da matriz
        cov2 = [[0.3, 1,1],[1,9,1], [1,1,9]]
        listaCov = funcoes.criarListas(cov1,cov2)
        -> exemplo (do professor): 
        covariancias = np.zeros((2,3,3)) 
        covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
        covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
        OBS: mesmo que os dados tenham médias ou covariâncias iguais, é preciso criar uma lista
        
        N = número de padrões que queremos gerar. 
       
        priors = uma lista de 1 dimensão com a probabilidade do padrão pertencer a cada classe
        -> exemplo:
        prior = np.array([1/2,1/2])
        
        plotar = true or false
        seed = controle do seed na geracao de dados aleatorios
        angulo = angulo da visualizacao em caso de plot 3d
      
    Saídas:
        dadossim = dados simulados no formato (caracteristicasxpadrões) para C classes
        -> ps. todas as classes tem a mesma matriz de covariância, mas médias deferentes
        -> fazer a transposta para deixar no formato (padrõesxcaracteristicas)
        Exemplo:
            dadossimT = dadossim.T
            classe1 = dadossimT[classessim==0,:]
            classe2 = dadossimT[classessim==1,:]
        
        classessim = classes dos dados simulados

Exemplo de como chamar a função:
dadossim, classessim = funcoes.gerandodadosgaussianos(listaMedias,listaCov,400, prior, plotar=True, seed=0,angulo=[20,120])
    '''        
    M,L=np.shape(medias)
    if (priors.all==0):
        priors=np.ones((M))/M
    if np.size(covariancias,axis=0)!=M |  np.size(covariancias,axis=1)!=L | np.size(covariancias,axis=2)!=L :
        print('Erro: confira a dimensao dos seus dados de input.')
        return    
    if np.size(priors)!=M :
        print('Erro: confira a dimensao dos priors.')
        return
    if np.sum(priors)!=1 :
        print('Erro: confira os valores dos priors.')
        return
    Ni=tuple(np.round(priors*N))
    np.random.seed(seed)      
    for i in range(M):
       if np.all(np.linalg.eigvals(covariancias[i]) > 0)==False :
           print('Erro: confira os valores da covariancia.')
       x=np.random.multivariate_normal(medias[i],covariancias[i],size=int(Ni[i])) 
       if i==0:
           dadossim=x.T
           classessim=np.zeros(int(Ni[i]),)
       else: 
           dadossim=np.concatenate((dadossim,x.T),axis=1)
           classessim=np.concatenate((classessim,np.zeros(int(Ni[i]),)+i),axis=0)

    if plotar: 
        if L==2: #2 caracteristicas, plot 2d
            plt.figure(figsize=(15,15))
            for i in range(M):                
                plt.plot(dadossim[0,classessim==i],dadossim[1,classessim==i],'o',fillstyle='none')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.show()
        elif L==3:
            plt.figure(figsize=(15,15))
            ax=plt.axes(projection='3d')
            for i in range(M):                
                ax.plot(dadossim[0,classessim==i],dadossim[1,classessim==i],dadossim[2,classessim==i],'o',fillstyle='none')
            ax.view_init(angulo[0],angulo[1])
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            plt.show()
        else:
            print('Grafico é exibido apenas para 2 ou 3 dimensões')
    return dadossim, classessim 

def criarListas(*args):
    '''
    Entradas:
    *args = infinitas entradas para coloca-las numa lista de facil acesso
    -> Importante: lista não é colocar um dado em cima do outro (isso seria concatenar)
    -> algumas funções (como a de gerar dados gaussianos, alguns classidicadores e FDA, só aceitam entrada nesse formato)

    Saídas:
    lista = lista(vetores / matrizes) com todas as entradas na mesma ordem
    '''

    c=np.shape(args)[0]
    lista=[]
    for i in range(c):
        lista.append(args[i])
    
    return lista

def calculaMedeCov(listaDados):
    #listaDados = lista com as matrizes de classe
    
    #saidas
    #medias = lista com a media de cada classe
    #covs = lista com a cov de cada classe
    #mediaT = media dos dados como um todo
    #covT = cov dos dados como um todo
    medias=[]
    covs=[]
    d=0
    for i in range(np.shape(listaDados)[0]):
        if (i==0):
            d=listaDados[i]
        else:
            d=np.concatenate((d,listaDados[i]),axis=0)
        medias.append(np.mean(listaDados[i],axis=0))
        covs.append(np.cov(listaDados[i], rowvar=False))
    
    mediaT=np.mean(d,axis=0)
    covT=np.cov(d, rowvar=False)
    return medias, covs, mediaT, covT

def separaClasses(dados,classes):
    """
    
    dados : matrix  NxA (padroes x qualquer coisa). (geralmente linhas de sinais e colunas de caracteristicas*) *ou tempo, ou qualquer coisa.
    classes : vetor N (padroes) com valores numericos que diz a qual classe cada padrão pertence. Vetor gabarito. 
     
    Returns
    -------
    classesSeparadas : um vetor cujos elementos são diversas matrizes, do tipo padrõesxcaracteristicas, um para cada classe. 
    Exemplo: [A,B,C,D] onde A é (pxc) da classe 1, B é (pxc) da classe 2 e assim por diante.
    valClasses : vetor de valores das respectivas classes
     
    Para usar essa saida e separar as classes de fato, após usar essa função, é preciso colocar saída uma em outra variavel. Exemplo:
          vigilia=classesSeparadas[0]
          rem1 =classesSeparadas[1]
        e assim por diante
        
    Exemplo de aplicação (aula 19)
    classesSeparadas, valClasses =funcoesprovav2.separaClasses(padroes, classes)

    classe1=classesSeparadas[0]
    classe2=classesSeparadas[1]

    print (np.shape(classe1))
    print (np.shape(classe2))
        
    """
    classesSeparadas=[]
    valClasses=valoresPossiveis(classes)
    
    for i in valClasses:
        matrixClasse=dados[classes==i,:]
        classesSeparadas.append(matrixClasse)
        
    return classesSeparadas,valClasses



################## FUNCOES DE PRE-PROCESSAMENTO

def encontraOutliers(padroes,p=3,method='desvio'):
    '''
     Encontra outlires baseando-se em dois métodos possiveis:
     method = 'desvio': mediana mais/menos p x desvio
     method = 'quartis': quartis  +/- p x intervalo entre quartis
     padroes = coluna de característica ( N x 1) (padrões x 1 caracteristica), já separado por classe
     p = numero de desvios ou de intervalos entre quartis a ser empregado 
     retorna lista com as posicoes dos outliers no array

    Entrada: padrões. É um vetor.
    -> as funções encontraoutliers e encontraoutliercomplot recebem apenas um vetor, mas função removeoutliers consegue receber vários. 
    Fazer classe por classe. 
    Saída:vetor que fala as casas dos outliers.

    '''

    if method =='desvio':
        md=np.median(padroes)
        std=np.std(padroes,ddof=1)
        th1=md+p*std
        th2=md-p*std
    elif method=='quartis':
        q3, q1 = np.percentile(padroes, [75 ,25])
        iqr=q3-q1
        th1=q3+p*iqr
        th2=q1-p*iqr
    outliers=(padroes>th1) | (padroes<th2)
    outs=[i for i, val in enumerate(outliers) if val]
    return outs

def encontraOutliersComPlot(padroes,p,method='desvio',Xlabel="",Ylabel=""):
    '''
  Encontra outlires baseando-se em dois métodos possiveis:
  metodo = 'desvio': mediana mais/menos p x desvio
  metodo = 'quartis': quartis  +/- p x intervalo entre quartis
  
    padroes = numpy array de uma característica ( N x 1) (lembrar de dar reshape(-1))
    No exercício, usamos p=3. 
    p = numero de desvios ou de intervalos entre quartis a ser empregado 
    Xlabel,Ylabel = legendas dos eixos x e y
   
    obs -> Fazer classe por classe. 
    Saída:retorna lista com as posicoes dos outliers no array
    Saída: plota imediatamente
    
    '''

    if method =='desvio':
        md=np.median(padroes)
        std=np.std(padroes,ddof=1)
        th1=md+p*std
        th2=md-p*std
    elif method=='quartis':
        q3, q1 = np.percentile(padroes, [75 ,25])
        iqr=q3-q1
        th1=q3+p*iqr
        th2=q1-p*iqr
    outliers=(padroes>th1) | (padroes<th2)
    outs=[i for i, val in enumerate(outliers) if val]
    
    x=np.arange(np.shape(padroes)[0])
    y=np.ones_like(x)
    
    plt.figure()
    plt.title('Detecção de outliers')
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.plot(x,y*th1,"r--",x,y*th2,"r--")
    plt.plot(padroes,"bo")
    plt.plot(outs,padroes[outs],"ro")
    plt.show ()
    return outs

def excluiOutliers(classe,p=3,metodo='desvio'):
    '''
    Remove os outliers do de uma matrix de classe.
    Retorna o mesmo conjunto de cados, mas sem os outliers
    Em vez de ser para (Nx1), é (NxA) onde A é o numero de caracteristicas
    
    Encontra outlires baseando-se em dois métodos possiveis:
    metodo = 'desvio': mediana mais/menos p x desvio
    metodo = 'quartis': quartis  +/- p x intervalo entre quartis
    
    p = numero de desvios ou de intervalos entre quartis a ser empregado 
    retorna o conjunto de dados(N-ixA) já sem os i outliers e o numero total de outliers removidos

Entrada: padrõesxcaracteristicas (linhas de sinais e colunas de caracteristicas)
Saída: padrõesxcaracteristicas sem os outliers
    '''
  
    l,c = np.shape(classe)
    outs=np.zeros(l)
    for i in range (c):
        outsColuna = encontraOutliers(classe[:,i],p,method=metodo)
        outs[outsColuna]=1
    classeNova=classe[outs==0,:]
    total=np.sum(outs)
    return classeNova,total

def normaliza(dados,metodo='linear',r=1):
    '''
    Entradas: 1 vetor de caracteristicas, do tipo (N,1). Faz a normalização dos dados N dessa única coluna.
    obs -> para normalizar um grupo, ver a função normaliza grupo. 
    
    dados = a coluna de uma caracteristica N x 1
    metodo ='linear' : normalizacao linear (padrao)
           = 'mmx': limitada entre -1 e 1
           = 'sfm': rescala nao linear no intervalo 0 a 1
    r = parametro do metodo sfm (padrao =1)
    a função retorna os dados normalizados
    
    Faz isso para todas as classes de uma vez. 
    
    Saída:vetor normalizado
    '''

    import numpy as np
    if metodo=='linear':
        M=np.mean(dados)
        S=np.std(dados,ddof=1)
        dadosnorm=(dados-M)/S
    elif metodo=='mmx':
        dadosnorm=2*dados/(np.max(dados)-np.min(dados))
        dadosnorm=dadosnorm - (np.min(dadosnorm)+1)
    elif metodo=='sfm':
        x=dados-np.mean(dados)
        x=-x/(r*np.std(dados))
        dadosnorm=1/(1+np.exp(x))  
        
    return dadosnorm

def normalizaGrupo(dados,metodo='linear',r=1):
    '''
    Realiza a normalizacao de um conjunto de dados 
    -> PS.NÃO separado em classe, as classes precisam estar todas juntas 
        A função normaliza grupo recebe só um conjunto de dados por vez.
        Portanto, além de deixar os dados no formado (padrõesxcaracteristcas)
        também é preciso concatenar as matrizes antes de aplicar os dados nela.
    
        Para concatenar: 
        matrizesConcatenadas,gabarito = funcoes.concatenarClasses(classe1,classe2)
        
        Para "desconcatenar", após a normalização:
        classe1 = dadosNormalizados[gabarito==0,:]
        classe2 = dadosNormalizados[gabarito==1,:]
        Para mais, ver item 1.c da atividade 10. 
        
    Entradas
    dados = uma ou mais matrizes N x L (padroesxcaracteriscas)
     metodo ='linear' : normalizacao linear (padrao)
            = 'mmx': limitada entre -1 e 1
            = 'sfm': rescala nao linear no intervalo 0 a 1
     r = parametro do metodo sfm (padrao =1)
    
    Saídas:
    A função retorna os dados normalizados
    '''

    import numpy as np
    dados2=np.zeros_like(dados)
    n,l=np.shape(dados)
    for i in range(l):
        dados2[:,i]=normaliza(dados[:,i],metodo,r)
        
    return dados2




##################  FUNCOES DE SELEÇAO DE CARACTERISTICAS

def TesteEstatisticoParaSelecaoDeClasses(dados1,dados2,alfa=0.05):
    '''
   Essa função determina as caracteristicas mais relevantes na separação de classes baseada em teste estatístico
   Ps. Se a normalidade da caracteristica for rejeitada, não usar. 
    
   Entrada:
       dados1 = matriz tipo (padroesxcaracteristicas) da classe 1 
       dados2 = matriz tipo (padroesxcaracteristicas) da classe 2
       alpha = taxa de erro tipo I do teste (erro tipo I é o falso positivo. tipo 2 é o falso negativo)
       
       
    Saída:
        rel = vetor com a significatividade. 
            diz quais são as posições das caracteristicas relevantes na separação de classes, por ordem. 
        p = vetor de p-values que diz o quão bom as caracteristicas são para separar a classes. 
            quanto MENOR, melhor é a caracteristica correspondente na separar as classes. 

    '''
    dados1=dados1.T
    dados2=dados2.T
    Ncarac,Npad=dados1.shape
    Ncarac2,Npad2=dados2.shape
    if Ncarac2!= Ncarac:
        print('Erro: matrizes devem ter o mesmo numero de caracteristicas!')
        return
    p=np.zeros(Ncarac)
    for i in range(Ncarac):
        s1=st.shapiro(dados1[i,:])
        s2=st.shapiro(dados2[i,:])
        if (s1[1]<0.05) | (s2[1]<0.05):
            res=st.ranksums(dados1[i,:],dados2[i,:])   
            p[i]=res.pvalue
            print('Aviso: normalidade rejeitada para a caracteristica nº '+ str(i+1))
        else:
            res=st.ttest_ind(dados1[i,:],dados2[i,:])
            p[i]=res.pvalue
    relevantes=(p<alfa)
    rel=[i for i,val in enumerate(relevantes) if val]
    return rel,p

def calculaRocAuc (classe1,classe2,plot=0):
    '''
    A ROC é uma curva cujo o valor da área embaixo dela (AUC) diz o quão bom uma caracteristica
    é para separar as classes.
    
   Entradas:
       classe1 = matriz (padrõesxcaracteristicas)
       classe = matriz (padrõesxcaracteristicas)
       plot = 0 ou 1 (não plotar e plotar)
       obs -> a função vai plotar um gráfico para cada caracteristica em ordem
       obs -> se forem 100 caracteristicas, serão 100 gráficos
       
   Saída:
       AUC = vetor que tem o mesmo número de casas que a quantidade de caracteristicas.
       Cada valor do vetor corresponde a AUC (área embaixo da curva ROC) da caracteristica respectiva.  
       Quanto mais próximo de 1, melhor é essa caracteristica para separar as classes. 
     '''

    import numpy as np
    from sklearn import metrics
    
    g1=np.zeros(classe1.shape[0])
    g2=np.ones(classe2.shape[0])
    
    dados = np.concatenate((classe1,classe2),axis=0) # axis =  0 para concatenar 1 em cima do outro
    g = np.concatenate((g1,g2))
    
    auc=np.zeros(np.shape(dados)[1])
    for i in range(np.shape(dados)[1]):
        auc[i]=metrics.roc_auc_score(g,dados[:,i])
        if(auc[i]<0.5):
            auc[i]=1-auc[i]
            tpr, fpr, thresholds=metrics.roc_curve(g,dados[:,i])
        else:
            fpr, tpr, thresholds=metrics.roc_curve(g,dados[:,i])
            
        if (plot==1):
            plt.figure()
            plt.plot(fpr,tpr,color="darkorange")
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC= %0.2f" % auc[i])
    return auc

def criterioFDRdeFisher(vet1, vet2):
    
    '''
    O critério FDR de fisher é utilizado para ver o quão boa uma caracteristica é na
    separação de classes. É tipo uma ROC, mas usando outro método (a distância média de duas classes de uma caracteristica)
    obs -> ficher é entreclasses/intraclasses
    
    Entrada:
        vet1 = classe1 (padrõesxcaracteristica1)
        vet2 = classe2 (padrõesxcaracteristica1)
        -> IMPORTANTE: são padroes por 1 (uma, one, una) caracteristica. 
        -> É preciso separar as caracteristicas para utilizar essa função. 
        -> Se eu quiser colocar + de uma caracteristica, usar a função selecaoEscalar e usar o critério fdr :)
        
    Saídas:
        fdr = critério de fisher entre os dois vetores calculados.
        -> é um número, quanto maior, melhor
    '''
    import numpy as np
    
    m1=np.mean(vet1)
    m2=np.mean(vet2)
    
    v1=np.var(vet1)
    v2=np.var(vet2)
    
    fdr=((m1-m2)**2)/(v1+v2)
    
    return fdr

def indCor(dados1,dados2):
    #ela é usada dentro da próxima, a sl_escalar
    #função interna desde arquivo
    t1=np.shape(dados1)[1]
    t2=np.shape(dados2)[1]
    cor=np.zeros((t1,t2))
    m1=np.mean(dados1,axis=0)
    m2=np.mean(dados1,axis=0)
    for i in range(t1):
        for j in range(t2):
            s1=dados1[:,i]-m1[i]
            s2=dados2[:,j]-m2[j]
            ss=np.sum(s1*s2)
            si=((np.sum(s1**2))*(np.sum(s2**2)))**(1/2)            
            a=ss/si
            cor[i,j]=a
    return cor

def selecaoEscalar(classe0,classe1,func="auc",pesoCriterio=0.5,k=2):
    '''
    Essa função ordena e seleciona as melhores caracteristicas utilizando um critério escalar a escolha (FDR (criterio de Fisher) ou AUC (ROC))
    
    Entradas:
        classe0 = (padrõesxcracteristicas) da classe1
        classe1 = (padrõesxcracteristicas) da classe2
        func = "auc" ou "fdr"
        pesoCriterio = valor de 0 a 1 que indica o peso do critério func na seleção da caracteristica
        -> se o pesoCriterio for 1, está ignorando a correçação.
        -> se o pesoCritério for 0, está ignorando o critério
        -> geralmente, usar algo entre 0.2 e 0.8. Por padrão está setado 0.5
        
       ATENÇÃO: as vezes o enunciado dá "peso correlação" em vez de peso critério. 
           -> pespCorrelação = 1-pesoCritério
           
        k = quantidade de caracteristicas que queremos escolher
    
    Saídas: vetor com o endereço das caracteristicas escolhidas
    '''

    dados = np.concatenate((classe0,classe1),axis=0)
    (L,N)=np.shape(dados)
    pesoCorrelacao=1-pesoCriterio
    pre_score=[]
    for i in range(N):
        if func=='fdr':
            score = criterioFDRdeFisher(classe0[:,i],classe1[:,i])
        elif func=='auc':
            auc=calculaRocAuc((classe0),(classe1))
            score=auc[N-1]
        pre_score.append(score)

    # return pre_score

    carac_escolhidas_index=[]
    index_select = pre_score.index(max(pre_score))
    carac_escolhidas_index.append(index_select)

    #caso k seja diferente de 1 iremos calcular para achar as n melhores features restantes
    for kn in range(1, k):
        #receberá o score de cada Mk para seleção da nova kn
        score_selecao = []
        for i in range(N):
            #calcula apenas para as features que não foram escolhidas
            if i not in carac_escolhidas_index:
                #levando em consideração o quão correlacionada/redundate a features testada no momento está em relação as já selecionadas
                correlacao = 0
                for index_carac in carac_escolhidas_index:
                    correlacao += abs(scipy.stats.pearsonr(dados[:,index_carac], dados[:,i])[0])
                score_ajustado = ((pesoCriterio * pre_score[i]) - pesoCorrelacao) * (correlacao/ len(carac_escolhidas_index))
                score_selecao.append(score_ajustado)
            else:
                #caso já tenhaos escolhido a feature colocamos um valor super negativo apenas
                #para garantir que essa n iremos selecionar novamente e que não precisemos
                #lidar com o shift do veto.
                #ex: começa com 4: 0 1 2 3
                #seleciona 1 ficaremos com: 0 1 2 (teriamos que mapear cada um
                # desses inices aos seus originais)
                score_selecao.append(-999999999)
        index_select = score_selecao.index(max(score_selecao))
        carac_escolhidas_index.append(index_select)

    return carac_escolhidas_index


def selecaoEscalarReserva(classe1,classe2,tipo="AUC",pesoCorrelacao=0.5,quantidade=2):
    '''
    Ignorar. 
    Tecnicamente funciona como a função seleçãoEscalar, mas os resultados não foram identicos ao da atividade 12. 
    Deixei aqui de reserva, caso seja útil para alguém no futuro.
    '''
    
    if(tipo=="AUC"):
        crit=calculaRocAuc(classe1,classe2)
    elif(tipo=="fisher"):
        crit=np.zeros((classe1.shape[1]))
        for i in range(classe1.shape[1]):
            crit[i]=criterioFDRdeFisher(classe1[:,i],classe2[:,i])
    
    dados = np.concatenate((classe1,classe2),axis=0)
    cov=np.cov(np.transpose(dados))
    #cov=np.corrcoef(np.transpose(dados))
    #cov=indCor(dados1, dados2)
    var=np.zeros((quantidade))
    for i in range(quantidade):
        if (i==0):
            critMax=crit.index(max(crit))            
            var[i]=critMax
        else:
            val=np.zeros((classe1.shape[1]))
            for j in range(classe1.shape[1]):
                sumCov=0                
                for k in range(i):
                    if(j!=int(var[k])):
                        sumCov+=cov[j,int(var[k])]                
                val[j]=(crit[j]*(1-pesoCorrelacao))-(pesoCorrelacao*sumCov/i)
            
            for k in range(i+1): 
                val[int(var[k])]=np.min(val)-1
            critMax=val.index(max(val))     
            var[i]=critMax
    return  var 


def selecaoVetorialExaustiva(listaClasses, K, criterio='J2'):
    '''
    Seleciona um grupo com as k melhores caracteristicas na separação de classes usando a seleção vetorial
    Entradas:
        listaClasses: lista do tipo (padrõesxcaracteristicas)(padroescaracteristicas)
        -> ps. fazer que listaClasses = [classe1,classe2]
        -> ps. ou usar a função criarListas
        -> IMPORTANTE. Essa função não aceita classes concatenadas (uma em cima da outra), tem que ser uma lista de matrizes (classes) mesmo
        
        K = número de caracteristicas que serão selecionadas
        criterio = criterio usado para calcular as melhores caracteristicas.
        criterio = pode ser J1, J2 ou J3.
        -> ps. Acho que o J1 está meio bugado. No exercício ele deu diferente do do professor. J2 e J3 deram iguais. 
        -> se possível, preferir usar o J2 ou J3
        
    Saídas:
        ordem = vetor com o endereço das caracteristicas escolhidas.
        maxcriterio = valor do critério escolhido (J1,J2,J3). O maior valor encontrado.
        -> quanto maior, melhor. E a função procura o maior maxcriterio possível
    '''
    
    L = listaClasses[0].shape[1] # numero de caracteristicas 
    M = len(listaClasses) # numero de classes
    
    Nc = np.zeros(M) # Numero de padroes em cada classes
    dados = listaClasses[0]
    for n in range(0, M):
        c = listaClasses[n]
        Nc[n] = c.shape[0]
        if n > 0:
            dados = np.concatenate((dados, c), axis=0)

    N = sum(Nc) # Total de padroes
    Pc = Nc/N  # Prob de padroes em cada classe
    
    maxcriterio = -np.inf
    for subset in combinations(range(0, L), K):
        Sw = np.zeros([K, K])
        #Sb = np.zeros([K, K])
        Sm = np.zeros([K, K])
        
        for n in range(0, M):
            c = listaClasses[n]
            matriz = np.cov(c[:, subset].T, ddof=0)
            Sw += Pc[n]*matriz
        
        Sm = np.cov(dados[:, subset].T, ddof=0)
        #Sb = Sm - Sw
        if criterio.upper() == 'J1':
            J1 = Sm.trace()/Sw.trace()
            if J1 > maxcriterio:
                maxcriterio = J1
                ordem = subset[:]
        elif criterio.upper() == 'J2':
            J2 = np.linalg.det(np.linalg.inv(Sw).dot(Sm))
            if J2 > maxcriterio:
                maxcriterio = J2
                ordem = subset[:]
        elif criterio.upper() == 'J3':
            J3 = (np.linalg.inv(Sw).dot(Sm)).trace()/K
            if J3 > maxcriterio:
                maxcriterio = J3
                ordem = subset[:]
    
    ordem = tuple(ordem)
    return ordem, maxcriterio

def fazPCA(matrizCarac,m):
 
  '''
  Tradução de PCA: análise de componentes principais.
  a PCA seleciona as melhores "caracteristicas" (em aspas porque a informação da caracteristica em si é perdida, devido a natureza da PCA).  
  o nome correto das "caracteristicas" seria componentes. 
  Ele mantém a informação, mas ele não se preocupa em separar os grupos. 
  As caracteristicas que a PCA seleciona não são as melhores para separar os dados, mas sim as que carregam mais informações (variância dos dados) 
  
    Entradas:
        matrizCarac = 1 matriz do tipo (padrõesxcaracteristicas)
        ps -> recebe apenas 1 matriz. Pode ser preciso concatenar as classes. 
        ps -> nesse caso, são dados concatenados mesmo. Não precisa usar a função "criarLista"
        ps -> usar a função concatenarClasses
        m = número de dimensões/quantidade de componentes que queremos
  
    Saídas:  
        autoValoresOrdenados = vetor com todos os autovalores da matriz de covariância, em ordem decrescente. 
        -> autovalores servirap para ordenar as caracteristicas da PCA
        autoVetoresOrdenados = matriz LxL com os respectivos autovetoes correspondentes aos autovalores
        -> é uma matriz usada para converter outros dados para a "dimensão" da PCA
        ->usar o .dot, igual na FDA, mas com o critério de quantidade ao final [:,0:m]
        -> exemplo:
            valoresPCA=valores.dot(autoValoresOrdenados)[:,0:m]
        
        matrizPCA = matriz (padrõesxcomponentes) com a projeção dos dados no mesmo espaço formamo pelas componentes (caracteristicas principais)
        -> 1º coluna é a 1º componente, 2º coluna é a segunda componente e assim por diante 
        -> para separar nas classes, usar que:
            classe1PCA = matrizPCA[classessim==0,:] #ele mantém as classes-gabarito de antes da transformação
            classe2PCA = matrizPCA[classessim==1,:]
        
        erro = % da quantidade de informações perdidas nos dados
        -> erro quadrático médio da projeção (em percentual da varicia total dos dados)
        -> valores referências: em 1D erro a partir de 50% é ruim (32% é ok para auxiliar na separação de classes) 
        -> valores referência: em 2D erro a partir de 16% é ruim (15% é ok para auxiliar na separação de classes)
        
     Essa função faz a diagnonalização da matriz de covariância e seleciona um determinado m (m<=quantidade de caracteristicas) deautovetores (que tem os maiores autovalores) para formar a nova base no espaço.     
        
  '''
  
  matriz=np.array(matrizCarac).T
  (l,n)=np.shape(matriz)

  matCovariancia = np.cov(matriz)
  autoValores,autoVetores = np.linalg.eigh(matCovariancia)

  autoVetoresOrdenados=np.zeros((l,l))
  autoValoresOrdenados=np.zeros(l)
  autoValores2=np.copy(autoValores)
  autoValores3=np.copy(autoValores)

  for i in range(l):
    max=np.max(autoValores2)

    valor=np.where(autoValores2==max)
    
    autoVetoresOrdenados[i,:]=np.array(autoVetores[:,valor]).T
    autoValoresOrdenados[i]=autoValores[valor]
    autoValores2[valor]=(np.min(autoValores2)-1)
    if i<m:
      autoValores3[valor]=0


  matrizPCAm = np.dot(autoVetoresOrdenados[:m,:],matriz)

  erro=(autoValores3.sum()/autoValores.sum())
  autoVetoresOrdenados=np.array(autoVetoresOrdenados).T
  matrizPCA=np.array(matrizPCAm).T
  return autoValoresOrdenados,autoVetoresOrdenados,matrizPCA,erro #matrizPCA seria equivalente aos novos dados na FDA

def SVD (matrizCarac,m):
  '''
  SVD - Decomposição em valores singulares 
  É equivalente ao PCA. Serve para reduzir a dimensionalidade dos dados
  
  Geralmente a SVD demora metade do tempo da PCA. 
  
    Entradas:
        matrizCarac = matriz (padroesxcaracteristicas)
        m = quantidade de dimensões/caracteristicas que queremos

    Saídas:
    u = autovetores da correlação de características
    s = autovalores
    vh = autovetores da correlação de padrões
    matriz2 = a matriz com a nova dimensionalidade

  '''
  matriz=np.array(matrizCarac).T
  u, s, vh = np.linalg.svd(matriz)
  comps=s.argsort()[::-1] #números das componentes em ordem decrescente dos seus autovalores
  comps=list(comps[0:m]) #lista com as primeiras "m" componentes
  autovetores2=u[:,comps] 
  autovetores2 = np.array(autovetores2).T
  autovalores_desc = np.delete(s, comps)
  matriz2 = np.dot(autovetores2,matriz)
  erro = (np.sum(autovalores_desc)) / (np.sum(s))
  matriz2=np.array(matriz2).T 
  return s, u, matriz2, erro

def autoordenar(eigvec, eigval):
    '''Usado na FDA'''
    # eigvalord = sorted(eigval, reverse=True)
    idx = np.argsort(eigval)
    eigvec = eigvec.T
    eigvecord = eigvec[idx]
    eigvalord = eigval[idx]
    eigvecord = eigvecord[-1::-1]
    eigvalord = eigvalord[-1::-1]
    return eigvecord.T, eigvalord

def scatter2(dados):
    '''Usado na FDA'''
    # Dados de treinamento (divididos em três classes)
    #X = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14]])
    #y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    #Calcula duas matrizes dentro de FDA (Fisher)
    
    X=0
    y=0
    for i in range(np.shape(dados)[0]):
        if (i==0):
            X=dados[i]
            y=np.zeros((np.shape(dados[i])[0]))
        else:
            X=np.concatenate((X,dados[i]),axis=0)
            y=np.concatenate((y,i*np.ones((np.shape(dados[i])[0]))))

    # Cálculo das médias das classes
    val=valoresPossiveis(y)
    mean_vectors = []
    for cl in val:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    
    # Cálculo da matriz de covariância dentro das classes
    S_W = np.zeros((np.shape(X)[1], np.shape(X)[1]))
    for cl,mv in zip(val, mean_vectors):
        class_sc_mat = np.cov(X[y==cl].T) # calcula a matriz de covariância da classe
        S_W += class_sc_mat
    
    # Cálculo da matriz de covariância entre as classes
    mean_overall = np.mean(X, axis=0)
    S_B = np.zeros((np.shape(X)[1], np.shape(X)[1]))
    for i,mean_vec in enumerate(mean_vectors):
        n = np.shape(X[y==i+1,:])[0]
        mean_vec = mean_vec.reshape(np.shape(X)[1],1) # reshaping
        mean_overall = mean_overall.reshape(np.shape(X)[1],1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    
    # matriz de fisher
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    
    # ordenando os autovetores pela ordem dos autovalores
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    
    return S_W,S_B

def fda(listaClasses,n):
    """
	O FDA muda a forma como "enxergamos" a amostra. 
    A FDA muda o angulo de onde observamos, como se estívessemos mudando o modelo
    FDA - Análise discriminatória de Fisher
 
    Entradas:
        classes: lista de classes (padrõesxcaracteristicas) **nao o gabarito. Usar a função "criar listas" 
        n: a quantidade de dimensões que eu quero que saia. Eu escolho. Nunca pode ser maior que o número de caracteristicas.
        n: número de caracteristicas que eu quero que saia após o FDA    
    
    Saídas:
        dadosPescados (piadinha com fisher) = os dados nos novos eixos
        autoVetores = são os novos autovetores. Serão usados para uma amostra nova de teste. 
    --->>>IMPORTANTE para fazer os dados de teste passarem pelo mesmo processo que os dados de treino. 
        COMO FAZER? Usar Y = X.dot(A)
        novodado=dadoantigo.dot(A) -> A é a saída mesmo. 
        
        A FDA também diminui o número de caracteristicas (definimos o quanto queremos ali na entrada n). Ela coloca o que considera mais importante. 
    
    EXEMPLO DE APLICAÇÃO: (aula 22) - COMO PLOTAR
    
    #Vamos separar os dados após o FDA
    classe1FDA=dadosPescados[classes==0,:]
    classe2FDA=dadosPescados[classes==1,:]
    print ("\n")
    print (np.shape(classe1FDA))
    print (np.shape(classe2FDA))

    #Agora plotando os gráficos:
    plt.figure()
    plt.title('FDA')
    plt.xlabel('padrões')
    plt.ylabel('caracteristica otimizada')
    plt.plot (classe1FDA[:,0],"bo")
    plt.plot (classe2FDA[:,0],"ro")
    plt.legend(loc="upper right") #para colocar a legenda
    plt.show ()
        
    """

    Sw, Sb = scatter2(listaClasses)
    eigval, eigvec = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

    eigvec, eigval = autoordenar(eigvec, eigval)

    C = len(listaClasses)  # Number of classes.
    autoVetores = np.abs(eigvec[:, 0:n])

    X = listaClasses[0]

    for n in range(1, C):
        X = np.concatenate((X, listaClasses[n]), axis=0)

    dadosPescados = X.dot(autoVetores)

    return dadosPescados,autoVetores




################## FUNCOES DOS CLASSIFICADORES

def whosMy(*args):
    #ela printa os dados q ela identificar sobre a variavel (tipo, formato, etc)
    #é basicamente uma função interna deste arquivo, usada em outras funções
  sequentialTypes = [dict, list, tuple] 
  for var in args:
    t=type(var)
    if t== np.ndarray:  
      print(type(var),var.dtype, var.shape)
    elif t in sequentialTypes: 
      print(type(var), len(var))
    else:
      print(type(var))

def classBayesMedCov (medias, covs, dTeste ,probs=0):
    '''
    #usar essa função quando eu já tiver as médias e covaciâncias do gabarito.
    -----
    medias = LISTA de médias do gabarito. São as médias das classes. 
    É a média de cada coluna das classes (como temos padrõesxcaracteristica, é a média de cada coluna de caracteristicas)
    se eu tiver 2 classes, essa lista terá 2 "linhas" e assim por diante. O mesmo para as covs.
    
    Usar a função "criar lista" para preparar essas entradas
    covs = LISTA de covariâncias do gabarito. 
    -----
    dTeste = dados que eu quero classificar
    caso o exercício não me dê os dados de teste, eu posso pegar uma "fatia" dos dados originais, que eu usei para treinar
    o classificador. por exemplo, 
    dTeste = (padroes[53:54,:]) #linha aleatória do conjunto original de dados
    
    eu também posso pegar linhas aleatórias com uma função. Exemplo:
    from sklearn.model_selection import train_test_split
    gabaritoTreino, gabaritoTeste, grupoTreino, grupoTeste = train_test_split(classes, padroes, train_size = 0.8, random_state = 46)
    46 é a quantidade de linhas que eu quero testar
    a coisa importante aqui é o grupoTeste
    dTeste = grupoTeste
    
    #probs = probabilidade do estar em cada classe (se forem 5 classes, será um vetor de 5)
    #é o prier
    #está relacionado ao tamanho da amostra
    #se o prier não é conhecido, podemos calcular ele proporcional a quantidade de amostras. 
    #eu posso colocar no prier diretamente o tamanho da classe 1 e da classe 2, pq a função normaliza.
    
    #-----
    #saídas:
    #P = distribuição de probabilidade. Linhas igual ao número de amostras e colunas o número de classes. Dará, para cada amostra, a chance de pertencer a cada classe.
    #resultadosTeste = vetor que diz qual classe pertence cada amostra. É o resultado. Começa em 0.  
    '''
    
    from scipy.stats import multivariate_normal
    cut=0
    if (type(dTeste)==list):
        dTeste=[dTeste,dTeste]
        cut=1
    dTeste=np.array(dTeste)
    P=np.zeros((np.shape(dTeste)[0],np.shape(medias)[0]))

    if (np.shape(probs)==np.shape(medias)):
        Pc=probs/np.sum(probs)
    else:        
        Pc=np.ones((np.shape(medias)[0]))/np.shape(medias)[0]       
    for i in range(np.shape(medias)[0]):
        m=medias[i]
        cv=covs[i]
        for j in range(np.shape(dTeste)[0]):
            x = dTeste[j,:]
            pj = multivariate_normal.pdf(x, mean=m, cov=cv)
            P[j,i] = pj*Pc[i]

    resultadosTeste = np.zeros((dTeste.shape[0]))
    
    for i in range(dTeste.shape[0]):        
        c = np.argmax(P[i,:])
        resultadosTeste[i]=c
        
    if(cut==1):
        P=P[0,:]
        resultadosTeste=resultadosTeste[0]
    return P,resultadosTeste

def valoresPossiveis(lista):
    #essa função recebe um vetor e retorna todos os valores desse vetor, ordenados e sem repetição
    #foi usada no classificador bayes
    possiveis=[]
    while(np.shape(lista)[0]>0):
        min=np.min(lista)
        possiveis.append(min)
        lista=lista[lista!=min]
    return possiveis

def classBayesTreino (dTreino, dTeste, gTreino):
    #Usar para dados que eu já tenho a classificação
    #essa função tenta classificar um grupo de dados tendo como base um grupo de mesmo formato já classificado
    #dTreino = matriz NxL (amostras x caracteriscas) já classificado entre as classes
    #dTeste = matriz MxL (amostras x caracteriscas) que será classificada 
    #gTreino = um vetor Nx1 (numero de amostras) que classifica os dados de dTreino
    #resultadosTeste = um vetor Mx1 (numero de amostras) com a suposta classificação de cada amostra do dTeste
    from scipy.stats import multivariate_normal
    P=np.zeros((dTeste.shape))
    grupos=valoresPossiveis(gTreino)
    Pc = np.zeros((np.shape(grupos)))
    for i in range(np.shape(grupos)[0]):
        elements = tuple(np.where(gTreino==grupos[i]))
        Pc[i] = (np.shape(elements)[1]/np.shape(gTreino)[0]) 
        Z = dTreino[elements,:][0]
        m = np.mean(Z, axis = 0)
        print("medias",np.shape(m))
        print(m)
        cv = np.cov(np.transpose(Z))
        print("cov",np.shape(cv))
        print(cv)
        
        for j in range(dTeste.shape[0]):
            x = dTeste[j,:]
            pj = multivariate_normal.pdf(x, mean=m, cov=cv)
            P[j,i] = pj*Pc[i]

    resultadosTeste = np.zeros((dTeste.shape[0]))
    
    for i in range(dTeste.shape[0]):        
        c = np.argmax(P[i,:])
        resultadosTeste[i]=grupos[c]
    return resultadosTeste

def classificadorDistancia(listaMedias, ListaCovs, dTeste):
    '''
    esse classificador dá ambos os resultados: para o classificador de distância mínima de Mahananobis e o euclidiano. 
    entradas: listaMedias - lista (cada casinha dele pode ter uma matriz) das classes. Gabarito.
    entradas: ListaCovs - igual ao item anterior, mas para as covariâncias
    dTest: padrões que queremos classificar.
        
    -------   
    saídas: rEucl é a classe que o dTeste pertence segundo o classificador euclidiano
    saídas: rMaha é a classe que o dTeste pertence segundo o classificador de Mahalanobis
    '''
    cut=0
    if (type(dTeste)==list):
        dTeste=[dTeste,dTeste]
        cut=1
    distMaha,rMaha=classMahaMedCov(listaMedias, ListaCovs, dTeste)    
    
    distEucl=cdist(dTeste,listaMedias)    
    rEucl=np.argmin(distEucl,axis=1)
    if(cut==1):
        distMaha=distMaha[0,:]
        rMaha=rMaha[0]
        distEucl=distEucl[0,:]
        rEucl=rEucl[0]
    
    return distEucl, distMaha, rEucl, rMaha

def perceptron(classe1, classe2, rho=0.1, niter=10000,plot=0):
    '''
    o perceptron não recebe um "dado" para ser classificado entre classes 1 ou 2. 
    ele apenas diz se as classes são separáveis ou não e encontra a linha que separa as duas classes se elas forem separáveis. 
    #Para plocar, usar plotSeparacaoW
    #Para classificar, usar o classificaPorW
    
    '''
        #input:
    #classe1 e classe2: matrizes 2d
    #classe1 e classe2 são entradas do tipo (caracteristicasxpadrões)
    #caso eu tenha algo em outro formato, seria bom alterar.
    #rho: Parâmetro de aprendizado. Pequeno: demora. Grande: pode não ser preciso.
    #niter:número máximo de passos. Valor em que o sistema para, mesmo que não seja o melhor resultado possível (tipo quando)
    #(cont.) o sistema é impossível
    
    #output:
    #w: o hiperplano que separa as classes - vetor com os pesos do classificador linear
    #os dois primeiros valores do vetor sao o peso. Se eles forem menores (em módulo) que o terceiro valor, então os pontos pertencem ao grupo 1
    #se eles forem maiores (em módulo), os pontos pertencem ao grupo 2
    #iter: quantos passos foram necessários para calcular (w)
    
    classe1=np.transpose(classe1)
    classe2=np.transpose(classe2)
    J = lambda p, w, dados, Y: sum(p[Y]*w.dot(dados[:, Y]))
    L, N1 = classe1.shape
    L2, N2 = classe2.shape
    iter = niter

    c = np.ones([1, N1+N2])
    c[0, N1:] = -1
    
    resultado="convergiu em"
    if L != L2:
        print('As classes precisam ter o mesmo numero de caracteristicas!')
        return
    dados = np.ones([L+1, N1+N2])
    dados[:-1, :N1] = classe1
    dados[:-1, N1:] = classe2

    w = np.random.randn(L+1)
    p = np.sign(w.dot(dados))
    
    inutil, Y = np.where(c != p)
    e = []
    erroAntes = np.zeros(L+1)
    erroSum = np.zeros(L+1)
    Kp = rho
    Ki = 0
    Kd = 0
    for n in range(0, niter+1):
        erro = sum((p[Y]*dados[:, Y]).T)
        P = Kp*erro
        I = Ki*(erroSum + erro)
        D = Kd*(erro - erroAntes)/0.4
        PD = P + I + D
        w = w - PD
        erroAntes = erro
        p = np.sign(w.dot(dados))
        e.append(J(p, w, dados, Y))
        inutil, Y = np.where(c != p)
        if np.where(c!=p)[0].shape[0] == 0:
            iter = n+1
            break
    if (n==niter):
        resultado="não converge com"
    if plot==1:
        
		#para o erro -> PARA HABILITAR: tirar as '#'
		#plt.figure() Gráfico do erro
		#plt.plot(e)
		#plt.title('Erro do perceptron')
		#plt.show()      
            
        xmin = np.min([np.min(classe1[0, :]), np.min(classe2[0, :])])
        xmax = np.max([np.max(classe1[0, :]), np.max(classe2[0, :])])
        #descobrir o mínimo e o máximo da caracteristica 'x' (horizontal no gráfico) 
        
        ymin = np.min([np.min(classe1[1, :]), np.min(classe2[1, :])])
        ymax = np.max([np.max(classe1[1, :]), np.max(classe2[1, :])])
        #descobrir o mínimo e o máximo da caracteristica 'y' (vertical no gráfico) 

        eixoX=np.linspace(xmin,xmax,1000) #vetor em x q vai do minimo ao maximo com 1000 pontos
        eixoY=(-w[2]-w[0]*eixoX)/w[1] #calculo do eixo "y", com base no x e no w obtido
        eixo=(eixoY>ymin)*(eixoY<ymax)#calcula os limites dos eixos
        eixoX=eixoX[eixo]#impoem os limites dos eixos
        eixoY=eixoY[eixo]
        
        plt.figure()
        plt.title('Perceptron - '+resultado+': '+ str(iter)+' iterações') #transformar o iter em string
        plt.plot(classe1[0,:],classe1[1,:],".b")
        plt.plot(classe2[0,:],classe2[1,:],".r")
        plt.plot(eixoX,eixoY,"b")

        plt.show()     

    return w, iter

def percepoket(classe1, classe2, rho=0.1, niter=10000, plot=0):
    """
    o perceptron não recebe um "dado" para ser classificado entre classes 1 ou 2.
    ele apenas diz se as classes são separáveis ou não e
    encontra a linha que separa as duas classes se elas forem separáveis. 
    input:
     classe1 e classe2: matrizes 2d
     classe1 e classe2 são entradas do tipo (caracteristicasxpadrões)
     caso eu tenha algo em outro formato, seria bom alterar.
     rho: Parâmetro de aprendizado. Pequeno: demora. Grande: pode não ser preciso.
     niter:número máximo de passos. Valor em que o sistema para, mesmo que não seja o melhor resultado possível
     (tipo quando o sistema é impossível)
    output:
     w: o hiperplano que separa as classes - vetor com os pesos do classificador linear
     os dois primeiros valores do vetor sao o peso. Se eles forem menores (em módulo) que o terceiro valor,
     então os pontos pertencem ao grupo 1
     se eles forem maiores (em módulo), os pontos pertencem ao grupo 2
     iter: quantos passos foram necessários para calcular (w)
    """
    classe1 = np.transpose(classe1)
    classe2 = np.transpose(classe2)
    J = lambda p, w, dados, Y: sum(p[Y] * w.dot(dados[:, Y]))
    L, N1 = classe1.shape
    L2, N2 = classe2.shape
    iter = niter

    c = np.ones([1, N1 + N2])
    c[0, N1:] = -1

    resultado = "convergiu em"
    if L != L2:
        print('As classes precisam ter o mesmo numero de caracteristicas!')
        return
    dados = np.ones([L + 1, N1 + N2])
    dados[:-1, :N1] = classe1
    dados[:-1, N1:] = classe2

    w = np.random.randn(L + 1)
    p = np.sign(w.dot(dados))

    inutil, Y = np.where(c != p)
    e = []
    erroAntes = np.zeros(L + 1)
    erroSum = np.zeros(L + 1)
    Kp = rho
    Ki = 0
    Kd = 0
    for n in range(0, niter + 1):
        erro = sum((p[Y] * dados[:, Y]).T)
        P = Kp * erro
        I = Ki * (erroSum + erro)
        D = Kd * (erro - erroAntes) / 0.4
        PD = P + I + D
        wp = w - PD
        erroAntes = erro
        e.append(J(p, wp, dados, Y))

        if J(p, wp, dados, Y) < J(p, w, dados, Y):
            w = wp

        p = np.sign(w.dot(dados))
        inutil, Y = np.where(c != p)
        if np.where(c != p)[0].shape[0] == 0:
            iter = n + 1
            break
    if n == niter:
        resultado = "não converge com"
    if plot == 1:
        # para o erro -> PARA HABILITAR: tirar as '#'
        # plt.figure() Gráfico do erro
        # plt.plot(e)
        # plt.title('Erro do perceptron pocket')
        # plt.show()

        xmin = np.min([np.min(classe1[0, :]), np.min(classe2[0, :])])
        xmax = np.max([np.max(classe1[0, :]), np.max(classe2[0, :])])
        # descobrir o mínimo e o máximo da caracteristica 'x' (horizontal no gráfico)

        ymin = np.min([np.min(classe1[1, :]), np.min(classe2[1, :])])
        ymax = np.max([np.max(classe1[1, :]), np.max(classe2[1, :])])
        # descobrir o mínimo e o máximo da caracteristica 'y' (vertical no gráfico)

        eixoX = np.linspace(xmin, xmax, 1000)  # vetor em x q vai do minimo ao maximo com 1000 pontos
        eixoY = (-w[2] - w[0] * eixoX) / w[1]  # calculo do eixo "y", com base no x e no w obtido
        eixo = (eixoY > ymin) * (eixoY < ymax)  # calcula os limites dos eixos
        eixoX = eixoX[eixo]  # impoem os limites dos eixos
        eixoY = eixoY[eixo]

        plt.figure()
        plt.title('Perceptron - ' + resultado + ': ' + str(iter) + ' iterações, p = ' + str(rho))  # transformar o iter em string
        plt.plot(classe1[0, :], classe1[1, :], ".b")
        plt.plot(classe2[0, :], classe2[1, :], ".r")
        plt.plot(eixoX, eixoY, "b")

        plt.show()

    return w, iter

def matmul(d1,d2):
    #função feita para LS
    mat=np.zeros((np.shape(d1)[0],np.shape(d2)[0]))
    for i in range(np.shape(d1)[0]):
        mat[i,:]=d1[i]*d2
    
    return mat

def ls(dados, gabarito, alpha=0):
    '''
    #Entradas:
        dados = são os dados no formato (padrõesxcaracteristicas) sem separação das classes (já que o gabarito também entra)
        -> recebe os dados concatenados. Classe 1 sobre a classe 2 ou até misturados. 
        -> atenção: não é uma lista. São os dados concatenados mesmo. 
        
        
        gabarito = é o vetor classes. Usamos que classes==valor da classe desejada.  
        -> exemplo:
            w2 = funcoes.ls(dadosT,classes==1)
            ou w2 = funcoes.ls(dadosT,classes==-1)
        
        alpha = peso. Parâmetro de regularização da inversão matricial. 
        -> ps. se der um erro de matriz única ao usar essa função, mudar o alpha. 
        -> ps. se eu não tiver informações sobre o alpha, deixar 0 mesmo. Ele afeta pouco. 
        
     Saídas:
         w = é um vetor de separação. 
         -> é análogo ao w do perceptron. 
         -> o que está acima do w pertence à uma classe e o que está abaixo pertence a outra
         -> inserir o w na função classificaPorW. Para plotar, usar a plotSeparacaoW
    
    #recebe os dados concatenados. Classe 1 sobre classe 2 ou até misturados.
    #ATENÇÃO: não é uma lista. É só os dados mesmo. 
    #(padrãoxcaracteristica)
    #Recebe também o vetor classes, que é o gabarito. (por isso recebe os dados juntos)

    '''
    soma=0
    c=(2*gabarito)-1
    s2=0
    for i in range(np.shape(dados)[0]):
        soma+=matmul(dados[i,:],dados[i,:])
        s2+=dados[i,:]*c[i]
    soma+(alpha*np.eye(np.shape(dados)[1]))
    soma=np.linalg.pinv(soma)
    w2 = soma.dot(s2)
    w=np.zeros(np.shape(w2)[0]+1)
    w[:-1]=w2
    return  w

def ls2(dados, gabarito, alpha=0):
    
    #Esse ls2 é um SVM "disfarçado". Dar preferência para usar o LS se o problema pedir. 
    #Entradas:
        #dados -> são os dados (padrõesxcaracteristicas) sem separação das classes (já que tb entre o gabarito)
    
    #O vetor gabarito tem que ser boleano. 
    #antes de chamar a função, fazer que gabarito==valor do gabarito na classe 1
    
    
    #recebe os dados concatenados. Classe 1 sobre classe 2 ou até misturados.
    #ATENÇÃO: não é uma lista. São só os dados mesmo. 
    #(padrãoxcaracteristica)
    #Recebe também o vetor classes, que é o gabarito. (por isso recebe os dados juntos)
    #alpha é um peso
    #alpha -> parâmetro de regularização da inversão matricial
    #pode dar um erro de matriz única ao usar essa função. Nesse caso, mudar o alpha
    #se eu não tiver informações sobre o alpha, ele é 0. Ele afeta pouco.
    
    
    #Saídas: w -> é um vetor de separação. 
    #é análogo ao w do perceptron. O que está acima do w pertence à uma classe e o que está abaixo pertence a outra
    #Uso o calcularperceptron para classificar o novo dado (por causa da saída w)
    #Para plocar, usar plotSeparacaoW

    from sklearn.svm import LinearSVC
    clf = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)

    clf.fit(dados, gabarito)
    w=np.zeros(np.shape(dados)[1]+1)
    w[:-1] = clf.coef_
    w[-1] = clf.intercept_
    return  w

def classificaPorW(dTeste,w): #antigo CalculaPerceptron ou LS. O que tem o w. 
    
    #dTeste = matriz MxL (amostras x caracteriscas) que será classificada
    #w = vetor de classificador linear (saida do perceptron)
    
    #resultadosTeste = um vetor Mx1 (numero de amostras) com a suposta classificação de cada amostra do dTeste 
    #se for true pertence à primeira classe
    #se for false, é da segunda classe
    #Para plotar, usar plotSeparacaoW
    '''
    Para contabilizar as amostras que pertecem ao grupo true e false
    tamanho = np.shape(resultadosTeste)
    grupo1 = np.sum(resultadosTeste) #ele só soma os trues
    grupo2 = (tamanho-grupo1)
    print ("\n")
    print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")
      
    '''
    
    cut=0
    if (type(dTeste)==list):
        dTeste=[dTeste,dTeste]
        cut=1
    t=np.shape(w)[0]
    dTeste=dTeste*w[:t-1]
    dTeste=np.sum(dTeste,axis=1)
    resultadosTeste=(dTeste+w[t-1])>0

    if(cut==1):
        resultadosTeste=resultadosTeste[0]
    
    return resultadosTeste

def plotSeparacaoW(dados,w):
    #usar para qualquer um q solte um W (perceptron e ls)
    '''
    Entradas:
        dados: dados (padrõesxcaracteristicas). Pode ser dadoTreino ou dadoTeste
        w: é a saída (classificador linear) do perceptron e LS. Se for true pertence a classe 1 e se for false, a classe 2. 
        
    Saídas:
        resultadosTeste==1
            
    '''
    if (np.shape(w)[0]==2):
        
        
        resultadosTeste = 1*classificaPorW(dados,w)
        
        d0=dados[resultadosTeste==0]
        d1=dados[resultadosTeste==1]
        
        shape=np.max((np.shape(d0)[0],np.shape(d1)[0]))
        x = np.linspace(0, shape-1,shape)
        if(w[0]==0):
            y=0
        else:
            y = -(w[1])/w[0]
        
        fig = plt.figure()
        plt.plot(x*y,"k-")
        plt.plot(d0,"bo")
        plt.plot(d1,"ro")
        
    elif (np.shape(w)[0]==3):
        
        x = np.linspace(np.min(dados[:,0]), np.max(dados[:,0]),100)        
        y = -(w[2]+x*w[0])/w[1]
        

        xl=((np.max(dados[:,0]-np.min(dados[:,0]))*0.1))
        yl=((np.max(dados[:,1]-np.min(dados[:,1]))*0.1))
        xg=((np.min(dados[:,0])-xl)<x)*(x<(np.max(dados[:,0])+xl))
        yg=((np.min(dados[:,1])-yl)<y)*(y<(np.max(dados[:,1])+yl))

        g=xg*yg

        resultadosTeste = 1*classificaPorW(dados,w)
        
        d0=dados[resultadosTeste==0]
        d1=dados[resultadosTeste==1]
        
        fig = plt.figure()
        plt.plot(x[g],y[g],"k-")
        plt.plot(d0[:,0],d0[:,1],"bo")
        plt.plot(d1[:,0],d1[:,1],"ro")
        
        
        
    elif (np.shape(w)[0]>3):
        x = np.linspace(np.min(dados[:,0]), np.max(dados[:,0]),100)
        y = np.linspace(np.min(dados[:,1]), np.max(dados[:,1]),100)
        x , y=np.meshgrid(x, y)
        z = -(w[-1]+w[0]*x+w[1]*y)/w[2]
        
        resultadosTeste = 1*classificaPorW(dados,w)
        
        d0=dados[resultadosTeste==0]
        d1=dados[resultadosTeste==1]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(d0[:, 0], d0[:, 1], d0[:, 2], c="blue")
        ax.scatter(d1[:, 0], d1[:, 1], d1[:, 2], c="red")
        ax.plot_surface(x, y, z,alpha=0.8)
        plt.show()
        
    return resultadosTeste==1

def classMahaMedCov(medias, covs, dTeste):
    #usado no classificador por distância
    from scipy.spatial import distance
    import scipy
    npX=dTeste
    npMEANS=medias
    npSIGMAS=covs

    npMAHALANOBIS = [[0 for _ in range(np.shape(npMEANS)[0])]
                for _ in range(np.shape(npX)[0])]
    for isample in range(np.shape(npX)[0]):
        for icluster in range(np.shape(npMEANS)[0]):
            npMAHALANOBIS[isample][icluster] = distance.mahalanobis(npX[isample],npMEANS[icluster],VI=scipy.linalg.pinv(npSIGMAS[icluster]))
    npMAHALANOBIS = np.array(npMAHALANOBIS)

    resultadosTeste = npMAHALANOBIS.argmin(axis=1)

    return npMAHALANOBIS, resultadosTeste

#SVM
"""
#copiar as linhas abaixo (que estão com #) e usar as duas próximas funções para usar o SVM
#from sklearn.svm import SVC #deixar aqui!
#svc=SVC(C=1.0, kernel="rbf", degree=3)

#-> dar outro nome para o svc se eu precisar usar mais de uma vez em outros tipos de dados (com novas classes, etc)
'''
C= 1 por padrão, usar valores mais altos para ignorar outliers e evitar overfit, porem valores muito mais altos podem causar underfit
kernel = tipo de curva que ele vai usar para tentar separar os dados 
    opções: 
        'linear’ = linear, o svm vai atuar como um perceptron
        ‘poly’ = polinomial, vai criar um polinomio do grau dado
        ‘rbf’ = não faço ideia mas é a padrão
        ‘sigmoid’ = não faço ideia
        ‘precomputed’ = não faço ideia
degree = usado apenas das kernel = 'poly', ele define o grau do polinomio que sera utilizado  
'''
"""
def preparaSVM(dadosTreino, svc, gabaritoTreino):
    #essa função treina o classificador SVM.
    
    #Entradas:
        #dados = dados de treino
        #gabaritoTreino = seriam as classes
        #svc = é uma "caixinha" onde "guardamos os dados treinados" 
        
    svc.fit(dadosTreino,gabaritoTreino)        
    return svc
    
def separaSVM(dados, svc, plot=0, gabarito=0):
    #Entrada:
        #dados = dados de teste ou dados que eu quero plotar
        #svc = saída do prepava SVM
        #gabarito = se eu souber as classes, posso inserir o vetor gabarito aqui. Ele não foi usado no classificador. Ai ele plota me mostrando o "certo" e o que ele classificou. 

    #Saídas:
        #Retorna um vetor dizendo se os dados inseridos pertencem a clásse 1 ou classe 2 de acordo com ele (não de acordo com o gabarito)


    resultadoTeste=svc.predict(dados)
    
    
        
    if (plot==1):
        
        x_min, x_max = dados[:, 0].min() - 1, dados[:, 0].max() + 1
        y_min, y_max = dados[:, 1].min() - 1, dados[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        
        if(type(gabarito)==int):
            if (gabarito==0):
                plt.figure()
                plt.plot(dados[:,0],dados[:,1],"ko")
                plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

        else:  
            d0=dados[gabarito==0,:]
            d1=dados[gabarito==1,:]        
    
            plt.figure()
            plt.plot(d0[:,0],d0[:,1],"bo")
            plt.plot(d1[:,0],d1[:,1],"ro")
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

    return resultadoTeste

################## FUNCOES/CONCEITOS PARA AVALIAR OS CLASSIFICADORES

def analiseEstatistica (real,teorico):
    '''
    Para avaliar o classificador, eu preciso saber a qual grupo a amostra de fato pertence
    e qual grupo o classificador indicou que ela pertence. 
    Importante: só serve para funções de duas classes, porque avalia uma como certa e outra errada. Ela não compararia entre duas "erradas"
    
    
    Entradas: 
        real = diz o grupo real que a amostra pertence. É o gabarito inteiro, mas usando == a classe que a amostra de fato pertence. 
        -> exemplo: classes==0 ou classes==1
        -> lembrar que classes é um vetor de 1 dimensão. Não é o classes1==0 (que seria padrõesxcaracteristicas)
       
        teorico = o grupo que o classificador disse que a amostra pertence.  
        -> exemplo: resultadoTeste==True
        também é um vetor de 1 dimensão.
        
        Exemplo:
            funcoes.analiseEstatistica(classes==1,resultadosTeste==True)
      
        
       Exemplo no k-fold:
            classesTeste==1, resultadosTeste==True

    '''
    vp=np.sum((real==True)*(teorico==True))
    vn=np.sum((real==False)*(teorico==False))
    fp=np.sum((real==False)*(teorico==True))
    fn=np.sum((real==True)*(teorico==False))
    
    sensibilidade = vp/(vp+fn)
    especificidade = vn/(vn+fp)
    precisao = vp/(vp+fp)
    prevalencia = (vp+fn)/(vp+vn+fp+fn)
    acuracia = (vp+vn)/(vp+vn+fp+fn)
    
    return vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia


""" Como utilizar o k-fold
Desvio baixo é bom, pois mostra que o sistema esta estavel (não esta com vies de underfit nem overfit)
#biblioteca -> from sklearn.model_selection import KFold 


Copiar e colar (e excluir as linhas com ->)
scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10) 
    -> #n_slits é a quantidade de folds
    -> ##shuffle = grupo de valores continuos ou misturados? False = continuos , True = misturados
    -> #random_state = "chave" do "parâmetro" aleatorio, se eu quiser grupos diferentes, é so mudar. 
    -> em geral, usar no random_state a quantidade de vezes que quero repetir a operação
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosTotais[iTreino,:], dadosTotais[iTeste,:]
        -> dadosTotais são os dados que eu quero usar para avaliar o classificador. 
        -> são do tipo (padrõesxcaracteristicas), sem separação de classes.
        -> se eu tiver os dados em classes separadas, terei que concatena-los. 
        
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]  
        ->classes é o vetor gabarito, que contém uma série de 0 e 1, ou 1 e 2, -1 e 1...
        -> não é obriatório usar, a não ser que o classificador utilize o gabarito para classificar. 
        
        --->>> [inserir aqui o classificador aqui com os parâmetros]<-----
        -> lembrando que os dados originais NÃO EXISTEM mais dentro do k-fold.
        -> a partir daqi, trabalhamos apenas co dadosTreino, dadosTeste, classeTreino e classeTeste
        -> classe1, classe2 será dadosTreino[classesTreino==1,:],dadosTreino[classesTreino==2,:] (ou ==0, ==-1, ==N)
        -> exemplo (Perceptron):
            w, iter = funcoes.percepoket(dadosTreino[classesTreino==1,:], dadosTreino[classesTreino==2,:],plot=0)
        -> exemplo (LS):
            w = funcoes.ls(dadosTreino, classesTreino==1)

       
       ----->[inserir o classifica por w, no caso do perceptron e LS]<---------
       -> o parametro será dadosTeste, que é a fatia que o k-fold elegeu para teste. 
       -> exemplo (Perceptron):
           resultadosTeste = funcoes.classificaPorW(dadosTeste, w)
        -> exemplo (LS)
        resultadosTeste = funcoes.classificaPorW(dadosTeste, w)
       
       
       ------>[inserir a análise estatistica/função que vê sensibilidade, vp, etc. ]<---------
       -> usar classesTeste==1, resultadosTeste==True
       -> exemplo (Perceptron):
           saídas = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        -> exemplo (LS):
            saídas = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        -> ps. saídas = vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia    
            
        
        fp=fp/np.sum(classesTeste==2)
        fn=fn/np.sum(classesTeste==1)
        scores.append((fp,fn))
        #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)
  

#print(np.shape(scores))
print ("\n")
print ("Aplicando k-fold no classificador perceptron")
print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))
#(Exemplo) printar resultado fazendo a média 

Para encontrar um exemplo, procurar no arquivo "#ExemploaquiK-fold"
"""


""" Como utilizar o k-fold + FDA
Copiar e colar (e excluir as linhas com ->)
scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10) 
    -> #n_slits é a quantidade de folds
    -> ##shuffle = grupo de valores continuos ou misturados? False = continuos , True = misturados
    -> #random_state = "chave" do "parâmetro" aleatorio, se eu quiser grupos diferentes, é so mudar. 
    -> em geral, usar no random_state a quantidade de vezes que quero repetir a operação
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosTotais[iTreino,:], dadosTotais[iTeste,:]
        -> dadosTotais são os dados que eu quero usar para avaliar o classificador. 
        -> são do tipo (padrõesxcaracteristicas), sem separação de classes.
        -> se eu tiver os dados em classes separadas, terei que concatena-los. 
        
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]  
        ->classes é o vetor gabarito, que contém uma série de 0 e 1, ou 1 e 2, -1 e 1...
        -> não é obriatório usar, a não ser que o classificador utilize o gabarito para classificar. 
        
        
        
        
        -----> FDA <----- A partir de agora, só usar dadosTreinoFDA e dadosTesteFDA, 
        ----> o gabarito (classesTreino e classsesTeste) permanecem com esse nome mesmo
        #-> uso a função separaclasses nos dados de treino 
        classe1Treino=dadosTreino[classesTreino==1]
        classe2Treino=dadosTreino[classesTreino==2]
        
        #-> uso a função criarListas para criar uma lista de classes como entrada da FDA
        
        listaClasses = funcoes.criarListas(classe1Treino,classe2Treino)
            
        #-> chamo a função FDA
        #exemplo:
        dadosTreinoFDA,autoVetores = funcoes.fda(listaClasses, 2)
        ->agora, multilpicar os dadosTeste pelos autovetores do FDA
        dadosTesteFDA=dadosTeste.dot(autoVetores)
        
        -----> FDA <-----
        
        
          
        --->>> [inserir aqui o classificador aqui com os parâmetros]<-----
        -> lembrando que os dados originais NÃO EXISTEM mais dentro do k-fold.
        -> a partir daqi, trabalhamos apenas co dadosTreino, dadosTeste, classeTreino e classeTeste
        -> classe1, classe2 será dadosTreino[classesTreino==1,:],dadosTreino[classesTreino==2,:] (ou ==0, ==-1, ==N)
        -> exemplo (Perceptron):
            w, iter = funcoes.percepoket(dadosTreinoFDA[classesTreino==1], dadosTreinoFDA[classesTreino==2],plot=0)
            
        -> exemplo (LS):
            w = funcoes.ls(dadosTreinoFDA, classesTreino==1)

       
       ----->[inserir o classifica por w, no caso do perceptron e LS]<---------
       -> o parametro será dadosTeste, que é a fatia que o k-fold elegeu para teste. 
       -> exemplo (Perceptron):
           resultadosTeste = funcoes.classificaPorW(dadosTesteFDA, w)
        -> exemplo (LS)
        resultadosTeste = funcoes.classificaPorW(dadosTesteFDA, w)
       
       
       ------>[inserir a análise estatistica/função que vê sensibilidade, vp, etc. ]<---------
       -> usar classesTeste==1, resultadosTeste==True
       -> exemplo (Perceptron):
           saídas = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        -> exemplo (LS):
            saídas = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        -> ps. saídas = vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia    
        -> classeTreino e classeTeste não ganham "FDA" no nome mesmo. 
    
   
    
        fp=fp/np.sum(classesTeste==2)
        fn=fn/np.sum(classesTeste==1)
        scores.append((fp,fn))
        #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)
  

#print(np.shape(scores))
print ("\n")
print ("Aplicando k-fold no classificador perceptron")
print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))
#(Exemplo) printar resultado fazendo a média 

Para encontrar um exemplo, procurar no arquivo "#ExemploaquiK-fold-FDA"
"""

#########################  EXERCICIOS DAS AULAS

"""Aula 6 - Probabilidade e inferência
arquivo = scipy.io.loadmat('Atividade 1.mat')
dados = arquivo['dados']
#print (np.shape(dados))

#dados = (4,200) (velocidade de condução nervosa x indivíduo)

linha1 = dados[0,:]
linha2 = dados[1,:]
linha3 = dados[2,:]
linha4 = dados[3,:]

mediaLinha1 = np.mean(linha1,axis=0) 
mediaLinha2 = np.mean(linha2,axis=0)
mediaLinha3 = np.mean(linha3,axis=0) 
mediaLinha4 = np.mean(linha4,axis=0)

varLinha1 = np.var(linha1)
varLinha2 = np.var(linha2)
varLinha3 = np.var(linha3)
varLinha4 = np.var(linha4)

print ("item A")
print ("A velocidade de condução média de cada medida (mCMAP, mSNAP, uCMAP e uSNAP são, respectivamente:", mediaLinha1, mediaLinha2, mediaLinha3, mediaLinha4)
print ("Já para as variâncias, os valores foram, respectivamente:",varLinha1,varLinha2,varLinha3,varLinha4)
print ("\n")

print ("item B")
print ("\n")

print ("item C")
print ("\n")

print ("item D")
print ("\n")

print ("item E")
print ("\n")

print ("Atividade 2: Inferência e diagnóstico")
print ("\n")


arquivo = scipy.io.loadmat('Atividade 2.mat')
dados = arquivo['dados']
#print (np.shape(dados))
#dados (19476,3)

coluna1 = dados[:,0] #resultados do teste PSA
coluna2 = dados[:,1] #resultados do teste DRE
coluna3 = dados[:,2] #resultados da biopsia - tomamos a biopsia como resultado verdadeiro. 

#item A
#sensibilidade: vp/(vp+fn)
#vp = deu 1 e é 1
#fn = deu 0 e é 1

l,c = np.shape(dados) #l:linhas, c:colunas
vp1 = (0) #deu 1 e é 1
fn1 = (0) #deu 0 e é 1
vp2 = (0) #deu 1 e é 1
fn2 = (0) #deu 0 e é 1

for i in range (l):
    if (coluna1[i]==coluna3[i]==[1]):
        vp1 = (vp1+1)
    elif (coluna1[i]==0 and coluna3[i]==1):
        fn1 = (fn1+1)
        
sensibilidade1 = (vp1/(vp1+fn1))

for i in range (l):
    if (coluna2[i]==coluna3[i]==[1]):
        vp2 = (vp2+1)
    elif (coluna2[i]==0 and coluna3[i]==1):
        fn2 = (fn2+1)
        
sensibilidade2 = (vp2/(vp2+fn2))
print ("item A")
print ("as sensibilidade dos exames PSA e DRE são, respectivamente:", (sensibilidade1*100),"% e",(sensibilidade2*100),"%")
print ("\n")

#item B
#especificidade: vn/(vn+fp)
#vn = deu 0 e é 0
#fp = deu 1 e é 0

l,c = np.shape(dados)
vn1 = (0)
fp1 = (0)
vn2 = (0)
fp2 = (0)

for i in range (l):
    if coluna1[i]==coluna3[i]==[0]:
        vn1 = (vn1+1)
    elif coluna1[i]== 1 and coluna3[1]==0:
        fp1 = (fp1+1)

especificidade1 = vn1/(vn1+fp1)

for i in range (l):
    if coluna2[i]==coluna3[i]==[0]:
        vn2 = (vn2+1)
    elif coluna2[i]== 1 and coluna3[1]==0:
        fp2 = (fp2+1)

especificidade2 = vn2/(vn2+fp2)
print ("item B")
print ("as específicidade dos exames PSA e DRE são, respectivamente:", (especificidade1*100),"% e",(especificidade2*100),"%")
print ("\n")

#item C
#Item C - Calcular a probabilidade de um paciente estar doente 
#Informações do exercício: o paciente testou positivo para PSA
#A prevalência da doença no grupo do paciente (homens 50+) é de 4,2%
#Probabilidade caso PSA dê positivo

'''
Raciocínio
P(T=1|D=1)=25,57% (sensibilidade)
P(T=0|D=1)=74,43%

P(T=0|D=0)=94,91% (especificidade)
P(T=1|D=0)=5,09%

P(D=1)=4,2% (prevalência) (ou 0.042)
P(D=0)=95,8%

Queremos P(D=1|T=1)
P(D=1|T=1)=(P(T=1|D=1)*P(D=1))/P(T=1)

Onde P(T=1) é P(T=1,D=1) + P(T=1,D=0)
Para encontrar as probabilidades conjuntas, usamos que: (P(x|y)=(P(x,y))/P(y))
Portanto, P(T=1) = P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)
Portanto, P(D=1|T=1) = (P(T=1|D=1)*P(D=1))/P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)
Portanto, P(D=1|T=1) = P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)
'''

probabilidade1 = ((sensibilidade1*0.042))/((sensibilidade1*(4.2/100))+(5.09/100)*95.8/100)

print ("item C: qual a probabilidade de um paciente estar doente sendo que o seu exame PSA deu positivo e a prevalência da doença para o seu grupo é de 4,2%?")
print ("a probabilidade é", probabilidade1*100,"%")
print ("\n")

#Probabilidade caso DRE dê positivo
'''
P(D=1|T=1) = P(T=1|D=1)P(D=1)+P(T=1|D=0)*P(D=0)

P(T=1|D=1)=17,75% (sensibilidade)
P(T=0|D=1)=(100-17.75)

P(T=0|D=0)=93,50 (especificidade)
P(T=1|D=0)=(100-93.50)

P(D=1)=4.2% (prevalência) (ou 0.042)
P(D=0)=95.8%

'''
probabilidade2 = ((sensibilidade2*0.042))/((sensibilidade2*(4.2/100))+((100-93.5)/100)*95.8/100)
print ("item C: qual a probabilidade de um paciente estar doente sendo que o seu exame DRE deu positivo e a prevalência da doença para o seu grupo é de 4,2%?")
print ("a probabilidade é", probabilidade2*100,"%")
"""

""" Aula 8 - Construção do espaço de caracteristicas
import numpy as np
import scipy
import matplotlib.pyplot as plt
import funcoes


dados = scipy.io.loadmat('Dados.mat')
sinal = dados['SINAL'] #trechosxtempo (1179x30)
estagios = dados['ESTAGIOS'].reshape(-1) #(1179x1)

#calcular as caracteristicas de cada um dos trechos do ECG

bandas = {'delta 1':[0.5,2.5],'delta 2':[2.5,4],'teta 1':[4,6],'teta 2':[6,8], 'alfa':[8,12],'beta':[12,20],'gama':[20,45]}
car,nomesc = funcoes.extraicarac(sinal, 100, bandas)
print (np.shape(car))
print (np.shape(nomesc))
print ("as caracteristicas calculadas foram:", nomesc)


#sinal trechosxtempo (1179x30)
#estagios (1179x1)
#car padrõesxcaracteristicas (1179x15)
#nomessc nome das caracteristicas (15)

classe1 = car[estagios==0] #vigilia
classe2 = car[estagios==1] #estagio1
classe3 = car[estagios==2] #estagio2
classe4 = car[estagios==3] #estagio3
classe5 = car[estagios==4] #estagio4
classe6 = car[estagios==5] #REM


#Plotar histogramas
plt.title('Média em vigilia')
plt.hist(classe1[:,0],rwidth=10, color='pink') #queremos plocar a coluna 1 da matriz classe(padrõesxcaracteristicas)
plt.show()  

#Plotar histogramas
plt.title('Variancia estágio 2')
plt.hist(classe2[:,1],rwidth=10, color='pink') #queremos plocar a coluna 1 da matriz classe(padrõesxcaracteristicas)
plt.show()

#Plotar histogramas
plt.title('Média estágio 1')
plt.hist(classe2[:,0],rwidth=10, color='pink') #queremos plocar a coluna 1 da matriz classe(padrõesxcaracteristicas)
plt.show()

#Construção do espaço de caracteristicas
#3 classes (classe1,classe5 e classe6)
#2 caracteristicas (complexidade e mobilidade) (3 e 2)

plt.figure()
plt.title('Espaço de caracteristicas')
plt.xlabel('Mobilidade')
plt.ylabel('Complexidade')
plt.plot (classe1[:,2],classe1[:,3],"bo") #queremos plotar as caracteristicas que estão nas colunas 2 e 3
plt.plot (classe5[:,2],classe5[:,3],"ro")
plt.plot (classe6[:,2],classe6[:,3],"go")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()


#Em 3D
#3 classes (classe1,classe5 e classe6)
#2 caracteristicas (mobilidade,f-central e delta 1) (2,4,8)

fig = plt.figure(figsize=(5,5))
plt.title('Espaço de caracteristicas')
ax = fig.add_subplot(111, projection='3d')

x1 = classe1[:,2] #queremos plotar as caracteristicas 2,4 e 8 da classe 1
y1 = classe1[:,4]
z1 = classe1[:,8]

x2 = classe5[:,2] #queremos plotar as caracteristicas 2,4 e 8 da classe 5
y2 = classe5[:,4]
z2 = classe5[:,8]

x3 = classe6[:,2] #queremos plotar as caracteristicas 2,4 e 8 da classe 6
y3 = classe6[:,4]
z3 = classe6[:,8]

ax.scatter(x1, y1, z1,color='pink',alpha=0.5)
ax.scatter(x2, y2, z2,color='blue',alpha=0.5)
ax.scatter(x3, y3, z3,color='green',alpha=0.5)
plt.show()
"""

""" Aula 10 - Outliers, normalização, pré seleção das melhores caracteristicas por teste estatístico e ROC
import numpy as np
import scipy
import matplotlib.pyplot as plt
import funcoes

#B - Encontrar outliers

dados = scipy.io.loadmat('Dados1') #Exemplo: 'Aula 10\Dados1.mat'
sinal = dados['sinal'].reshape(-1)
print (np.shape(sinal))
#sinal(300)

#metodo: desvio
outs = funcoes.encontraOutliersComPlot(sinal, 3, method='desvio',Xlabel="amostras",Ylabel="" )

#método: quartis
outs = funcoes.encontraOutliersComPlot(sinal, 3, method='quartis',Xlabel="amostras",Ylabel="" )

#C - Testar a normalização de dados

dados = scipy.io.loadmat('Dados2') #Exemplo: 'Aula 10\Dados1.mat'
#print (dados)
med = dados['med']
ske = dados['ske']
#print (np.shape (med))
#print (np.shape (ske))

#med (5x2) (padrõesxclasse)
#ske (5,2) (padrõesxclasse)

#Eu prefiro (e criei todas as minhas funções) pensando em receber o formato 
#classe (padrõesxcaracteristicas) em vez de caracteristica (padrõesxclasse)

#Portanto, para continuar a questão, vamos converter as entradas para o formato desejado. 
#Usar a função 

matrizes = funcoes.matrizClasseParaMatrizCaracteristica(med,ske)
classe1 = matrizes[0] #grau1
classe2 = matrizes[1] #grau2

#print (np.shape (classe1))
#print (np.shape (classe2))

#med (5x2) (padrõesxclasse)
#ske (5,2) (padrõesxclasse)
#matrizes ()
#classe1 (5,2) (padrõesxcaracteristicas)
#classe1 (5,2) (padrõesxcaracteristicas)

#Então vamos normalizar as classes. 
#A função normaliza grupo recebe dados concatenados. 
#Portanto, primeiro vamos concatenar as classes.


classesConcatenadas,gabarito = funcoes.concatenarClasses(classe1, classe2)

classesConcatenadasNormalizadas = funcoes.normalizaGrupo(classesConcatenadas,metodo='linear',r=1)


#para "desconcatenar"

classe1Normalizada = classesConcatenadasNormalizadas[gabarito==0,:]
classe2Normalizada = classesConcatenadasNormalizadas[gabarito==1,:]
       
#vamos plotar a obliquidade pela média no espaço de caracteristcas para ver os resultados da normalização. 
#1º caracteristica: med. 2º: ske

plt.figure()
plt.title('Sem normalização')
plt.xlabel('Ske')
plt.ylabel('Média')
plt.plot (classe1[:,0],classe1[:,1],"bo") #queremos plotar as caracteristicas que estão nas colunas 2 e 3
plt.plot (classe2[:,0],classe2[:,1],"ro")
plt.show ()

plt.figure()
plt.title('Com a normalização')
plt.xlabel('Ske')
plt.ylabel('Média')
plt.plot (classe1Normalizada[:,0],classe1Normalizada[:,1],"bo") #queremos plotar as caracteristicas que estão nas colunas 2 e 3
plt.plot (classe2Normalizada[:,0],classe2Normalizada[:,1],"ro")
plt.show ()

#C Aplicar um teste estatistico para descobrir as caracteristicas mais relevantes na separação de classes
#Essa função recebe algo do

rel,p = funcoes.TesteEstatisticoParaSelecaoDeClasses(classe1Normalizada,classe2Normalizada)

print ('rel:')
print (rel)
print ('p-value:')
print (p)

print ('segundo o rel (0), a caracteristica 1 é mais relevante')
print ('o p-value é menor para a caracteristica 1, portanto ela é melhor em separar as classes')

##############################
#Curva ROC
auc = funcoes.calculaRocAuc(classe1Normalizada, classe2Normalizada, plot=1)
"""


""" Aula 12 - seleção escalar (AUC(ROC) e FDR(Critério de Fisher))

import numpy as np
import scipy
import scipy.io
import scipy.stats
import funcoes


arquivo = scipy.io.loadmat('Dadosaula12')
classe1 = arquivo['figadoadiposo']
classe2 = arquivo['figadocirrotico']
#print (np.shape(classe1))
#print (np.shape(classe2))

#classe1 (10x4) (padrõesxcaracteristicas) figado adiposo
#classe2 (10x4) (padrõesxcaracteristicas) figado cirrotico
#caracteristicas: media, desvio padrão, obliquidade e curtose

#Escolher 2 melhores caracteristicas das 4 dadas usando seleção escalar (AUC e FDR)


melhoresCaracFDR1 = funcoes.selecaoEscalar(classe1, classe2, func="fdr",pesoCriterio=0.2,k=2)
print (melhoresCaracFDR1)
print ("com FDR e peso 0.8, as melhores caracteristicas foram")
print ("media e curtose")
print ("devia ser med e curtose")

melhoresCaracFDR2 = funcoes.selecaoEscalar(classe1, classe2, func="fdr",pesoCriterio=0.5,k=2)
print (melhoresCaracFDR2)
print ("com FDR e peso 0.5 (igual entre correlação e critério), as melhores caracteristicas foram")
print ("media e desvio padrão")
print ("devia ser med e desvio padrão")

melhoresCaracAUC1 = funcoes.selecaoEscalar(classe1, classe2, func="auc",pesoCriterio=0.2,k=2)
print (melhoresCaracAUC1)
print ("com FDR e peso 0.8, as melhores caracteristicas foram")
print ("media e curtose")
print ("devia ser med e curtose")


melhoresCaracAUC2 = funcoes.selecaoEscalar(classe1, classe2, func="auc",pesoCriterio=0.5,k=2)
print (melhoresCaracAUC2)
print ("com FDR e peso 0.8, as melhores caracteristicas foram")
print ("media e desvio curtose")
print ("devia ser med e curtose")

"""
""" Aula 14 - seleção vetorial
import numpy as np
import scipy.io
import scipy.stats
import funcoes

arquivo = scipy.io.loadmat('Dadosaula12')
classe1 = arquivo['figadoadiposo']
classe2 = arquivo['figadocirrotico']
#print (np.shape(classe1))
#print (np.shape(classe2))

#classe1 (10x4) (padrõesxcaracteristicas) figado adiposo
#classe2 (10x4) (padrõesxcaracteristicas) figado cirrotico
#caracteristicas: media, desvio padrão, obliquidade e curtose


#Normalização primeiro 
classesConcatenadas,gabarito = funcoes.concatenarClasses(classe1,classe2)
classesConcatenadasNorm = funcoes.normalizaGrupo(classesConcatenadas)

#Para "desconcatenar", após a normalização:
classe1Norm = classesConcatenadasNorm[gabarito==0,:]
classe2Norm = classesConcatenadasNorm[gabarito==1,:]

#Usando seleção vetorial
listaClasses = [classe1,classe2]

ordem, maxcriterio = funcoes.selecaoVetorialExaustiva(listaClasses,2)
print (ordem)
print (maxcriterio)

print ("As caracteristicas selecionadas pela seleção vetorial com J1 foram")
print ("media e obliquidade")
print ("devia ser med e desvio")
"""

""" Aula 16 - PCA
import numpy as np
import scipy
import matplotlib.pyplot as plt
import funcoes

#Gerando dados gaussianicos 
#Eu quero: 2 classes, 400 padrões (200 por classe) e 3 caracteristicas. 

#médias de cada classe
media1 = (-2,6,6)
media2 = (2,6,6)

#covariancias para cada classe
#como a média tem 3 parâmetros, a covariância será 3x3
cov1 = [[0.3, 1,1],[1,9,1], [1,1,9]] #criando a matriz. cada [] é uma linha da matriz
cov2 = [[0.3, 1,1],[1,9,1], [1,1,9]]

prior = np.array([1/2,1/2])

listaMedias = funcoes.criarListas(media1,media2)
listaCov = funcoes.criarListas(cov1,cov2)

dadossim, classessim = funcoes.gerandodadosgaussianos(listaMedias,listaCov,400, prior, plotar=False, seed=0,angulo=[20,120])
dadossimT = dadossim.T
classe1 = dadossimT[classessim==0,:]
classe2 = dadossimT[classessim==1,:]

#dadossim são dados simulados no formato (caracteristicasxpadrões) para C classes (3,400)
#dadossim.T são dados simulador no formato (padrõesxcaracteristicas) (400,3)
#classessim - classes dos dados simulados
#classe1 (200,3)
#classe2 (200,3)

#Computando PCA em 1D
#concatenando as classes

classesConcatenadas,gabarito = funcoes.concatenarClasses(classe1,classe2)
autoValoresOrdenados,autoVetoresOrdenados,matrizPCA,erro = funcoes.fazPCA(classesConcatenadas, 1)
print ("o erro em 1 dimensão foi de", erro*100,"%")
#print (autoValoresOrdenados)
#print (autoVetoresOrdenados)
#print (matrizPCA)

#matrizPCA é uma matriz do tipo (padrõesxcomponentes)

classe1PCA = matrizPCA[classessim==0,:]
classe2PCA = matrizPCA[classessim==1,:]
#print (np.shape(classe1PCA))
#print (np.shape(classe2PCA))


plt.figure()
plt.title('PCA 1D')
plt.xlabel('')
plt.ylabel('')
plt.plot (classe1PCA[:,0],"bo")
plt.plot (classe2PCA[:,0],"ro")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()


#Computando PCA em 2D
#concatenando as classes

classesConcatenadas,gabarito = funcoes.concatenarClasses(classe1,classe2)
autoValoresOrdenados,autoVetoresOrdenados,matrizPCA2,erro = funcoes.fazPCA(classesConcatenadas, 2)
print ("o erro em 2 dimensão foi de", erro*100,"%")
#print (autoValoresOrdenados)
#print (autoVetoresOrdenados)
#print (matrizPCA)

classe1PCA2 = matrizPCA2[classessim==0,:]
classe2PCA2 = matrizPCA2[classessim==1,:]

plt.figure()
plt.title('PCA 2D')
plt.xlabel('')
plt.ylabel('')
plt.plot (classe1PCA2[:,0],classe1PCA2[:,1],"bo")
plt.plot (classe2PCA2[:,0],classe2PCA2[:,1],"ro")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()

####### SVD x PCA
"""

""" Aula 17 - Classificador Bayesiano x Mahalanobis x Euclidiano
import numpy as np
import scipy
import matplotlib.pyplot as plt
import funcoes


#Gerando dados gaussianos
medClasse1 = (0,0,0)
medClasse2 = (0.5,0.5,0.5)
listaMed = funcoes.criarListas(medClasse1,medClasse2)

cov1 = [[0.8, 0.01, 0.01],[0.01,0.2, 0.01], [0.01,0.01,0.2]]
cov2 = cov1
listaCov = funcoes.criarListas(cov1,cov2)
priors=np.array([1/2,1/2])

dadossim, classessim = funcoes.gerandodadosgaussianos(listaMed, listaCov, 600, priors,plotar=False)

dadossimT = dadossim.T #para obter os dados no formato (padrãoxcaracteristicas)
classe1 = dadossimT[classessim==0,:]
classe2 = dadossimT[classessim==1,:]

#print (np.shape(dadossimT))
#print (np.shape(classe1))
#print (np.shape(classe2))

#dadossimT (padrãoxcaracteristica)
#classe1 (padrãoxcaracteristica)
#classe2 (padrãoxcaracteristica)
#medClasse1
#medClasse2
#listaMed lista de médias
#cov1
#cov2
#listaCov lista de covariancias
#x - dado que queremos classificar



dTeste = [0.1,0.5,0.1]
#print (np.shape(dTeste))


fig = plt.figure(figsize=(5,5))
plt.title('Dados simulados e x')
ax = fig.add_subplot(111, projection='3d')

x1 = classe1[:,0]
y1 = classe1[:,1]
z1 = classe1[:,2]

x2 = classe2[:,0] #queremos plotar as caracteristicas 2,4 e 8 da classe 5
y2 = classe2[:,1]
z2 = classe2[:,2]

x3 = dTeste[0] #dado que eu quero plotar (x)
y3 = dTeste[1]
z3 = dTeste[2]


ax.scatter(x1, y1, z1,color='pink',alpha=0.2)
ax.scatter(x2, y2, z2,color='pink',alpha=0.2)
ax.scatter(x3, y3, z3,color='black',alpha=1)
plt.show()


#Classifique dTeste usando a estratégia Bayesiana
P,resultadosTeste = funcoes.classBayesMedCov (listaMed, listaCov, dTeste ,probs=0)

#print (resultadosTeste)
print ("Pertence a classe 2 segundo a estratégia Bayesiana")

#Classifique usando de distância mínima de euclidiana
distEucl, distMaha, rEucl, rMaha = funcoes.classificadorDistancia(listaMed, listaCov, dTeste)
#print (rEucl)
#print (rMaha)
print ("Pertence a classe 1 segundo a o classificador euclidiano")
print ("Pertence a classe 2 segundo a o classificador mahalanobis")

#ITEM E - Comente os resultados dos itens b-d
print ("\n")
print ("Classificadores bayesianos versus de distância mínima")
print ("O classificador que utiliza a estratégia bayesiana baseia-se no cálculo da probabilidade de certo dado pertencer à determinada classe, enquanto os classificadores de distância mínima fazem uma análise geométrica dos padrões, de acordo com suas posições no espaço de caracteristicas, partindo da ideia de que pontos mais próximos pertencem a mesma classe.")
print ("\n")
print ("Os classificadores bayesianos são mais adequados para conjuntos complexos, por levarem em consideração a probabilidade a priori. Eles também são mais adequados para grandes conjuntos de dados, uma vez que os classificadores de distância mínima podem 'se confundir' se houverem muitas classes. Seguindo a mesma lógica, os bayesianos são melhores para dados de alta dimensionalidade")
print ("\n")
print ("Classificadores de distância mínima: distância euclidiana versus mahalanobis:")
print ("A distância euclidiana calcula a distância mais curta entre dois pontos seguindo uma  linha reta, enquanto a distância de Mahalanobis leva em consideração a variância e a covariância dos dados, medindo a distância de um ponto até a distribuição de dados")
print ("\n")
print ("A distância euclidiana funciona bem em conjunto de dados que tem variâncias semelhantes (enquanto a distância de Mahalanobis pode ser vantajosa em conjunto de dados com variâncias diferentes, mas é mais sensível a outliers)")
print ("\n")
print ("Nesse caso, como não temos outliers, embora os dados tenham a mesma covariância, é mais provável que eles pertencam à classe 2 - uma vez que esse foi o resultado dos classificadores de bayesiano e de mahalanobis, que são mais complexos que o euclidiano e respectivamente mais 'robustos'")
"""

""" Aula 19 - classificadores bayesianos e de Mahalanobis. Dados dado pelo enunciado. Cálculo da sensibilidade, etc. 

dados = scipy.io.loadmat('Dados.mat')
padroes = dados['padroes']
classes = dados['classes'].reshape(-1)
print (np.shape(padroes))
print (np.shape(classes))
#Temos 1600 linhas de padrões e 4 colunas de caracteristicas

#A) - Estimar média e cov das classes, aplicar um classificador bayesiano e verificar a sensibilidade, específicidade e acurácia do classificador
possiveis = funcoesprovav2.valoresPossiveis(classes)
print (possiveis)
#E as classes estão denominadas "-1" e "1". 

#Separando as classes
classe1=padroes[classes==-1,:]
classe2=padroes[classes==1,:]
print (np.shape(classe1))
print (np.shape(classe2))

#Agora, vamos calcular as médias e as covariâncias de cada classe.
#A classe 1 tem dimensões 1000x4 (padrõesxcaracteristicas)
#e a classe 2 tem dimentões 600x4 (padrõesxcaracteristicas)
#queremos saber a média de cada caracteristica, isto é, a média de cada coluna.
#por isso, usamos o argumento "axis" 

#medias
mediaClasse1 = np.mean(classe1,axis=0)
mediaClasse2 = np.mean(classe2,axis=0)

print ("média 1")
print (mediaClasse1)
print ("média 2")
print (mediaClasse2)
print ("\n")

#isso gerou 2 vetores de 4 valores, sendo cada valor a média de uma caracteristica. 

#Calculando a covariância das classes
#Mesma lógica da média, calcular a cov das colunas
cov1 = np.cov(classe1, rowvar=False) #para calcular a cov nas colunas
cov2 = np.cov(classe2, rowvar=False) #para calcular a cov nas colunas

print ("covariância 1")
print (cov1)
print ("covariância 2")
print (cov2)


#################################################
#ITEM 1
#Classificador bayesiano. 
#média = vetor das médias das colunas das classes 1 e 2
#covs = matriz das covariâncias das colunas das classes 1 e 2
#dTest = pontos (linhas) aleatórias que pertencem a esse grupo. Nesse caso, 
#como o enunciado não me dá o dado de teste, posso pegar aleatório dos outros.
#para obter o dTeste, vamos usar uma função que pega dados aleatórios do grupo original (trem de teste)

medias = funcoesprovav2.criarListas(mediaClasse1,mediaClasse2)
#print (np.shape(medias))
covs = funcoesprovav2.criarListas(cov1,cov2)

#dTeste
from sklearn.model_selection import train_test_split
gabaritoTeste, gabaritoTreino, grupoTeste, grupoTreino = train_test_split(classes, padroes, train_size = 100, random_state = 46)
#a coisa importante aqui é o grupoTeste
dTeste = grupoTeste
print(np.shape(dTeste))
#Finalmente, chamando o classificador: 
P,resultadosTeste = funcoesprovav2.classBayesMedCov(medias,covs,dTeste,probs=0)
print ("\n")
#print (P)
print ("resultados")
#Agora vamos calcular a sensibilidade,especificidade,precisão e prevalencia do classificador
#utilizando a função analiseEstatistica

sensibilidade,especificidade,precisão,prevalencia,acuracia = funcoesprovav2.analiseEstatistica((gabaritoTeste==-1),(resultadosTeste==0))
print ("sensibilidade:", sensibilidade)
print ("especificidade:",especificidade)
print("acuracia",acuracia)


#######################################################
#ITEM 2 
#Classificador de distância mínima Mahalanobis

#medias - do item anterior
#covs - do item anterior
#Dtest
gabaritoTreino, gabaritoTeste1, grupoTreino, grupoTeste1 = train_test_split(classes, padroes, train_size = 0.8, random_state = 46)
#a coisa importante aqui é o grupoTeste
dTeste1 = grupoTeste1

distEucl, distMaha, rEucl, rMaha = funcoesprovav2.classificadorDistancia(medias,covs,dTeste1)
#print (rMaha)

sensibilidade1,especificidade1,precisão,prevalencia,acuracia1 = funcoesprovav2.analiseEstatistica((gabaritoTeste1==-1),(rMaha==0))
print ("resultados")
print ("sensibilidade:", sensibilidade1)
print ("especificidade:",especificidade1)
print("acuracia",acuracia1)

print ("Comparando")
print ("sensibilidade do classificador bayesiano e de mahananobis, respectivamente:", "\n", sensibilidade, "e", sensibilidade1)
print ("específicidade do classificador bayesiano e de mahananobis, respectivamente:","\n", especificidade, "e", especificidade1)
print ("acurácia do classificador bayesiano e de mahananobis, respectivamente:","\n", acuracia, "e", acuracia1)

###########################
#ITEM 3
#Interpretar os resultados
print("\n")
print ("O classificador bayseano se mostrou superior ao de distância mínima de Mahalanobis".
"""

""" - Aula 22 - Perceptron e LDA (FDA - Fisher)
dados = scipy.io.loadmat('dadosex2.mat')
classe1 = dados['Classe1']
classe2a = dados['Classe2a']
classe2b = dados['Classe2b']
classe2c = dados['Classe2c']
classe2d = dados['Classe2d']

print (np.shape(classe1))
print (np.shape(classe2a))
print (np.shape(classe2b))
print (np.shape(classe2c))
print (np.shape(classe2d))

#Ao printar, nota-se que o tamanho dos arquivos é (2x100). 
#Isto é, está no formato caracteristicaxpadrões
#Para colocar no formato padrõesxcaracteristica (trocar linhas e colunas)
#vamos utilizar a transposta. 


classe1T = np.transpose(classe1)
classe2aT = np.transpose(classe2a)
classe2bT = np.transpose(classe2b)
classe2cT = np.transpose(classe2c)
classe2dT = np.transpose(classe2d)

print ("\n")
print (np.shape(classe1T))
print (np.shape(classe2aT))
print (np.shape(classe2bT))
print (np.shape(classe2cT))
print (np.shape(classe2dT))

#A
#velocidade do algorítimo para separar as classes 1 e 2A, 2B, 2C e 2C com paramtro de aprendizagem = 0.05
#Classe 1 e classe 2A

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2aT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2A foi de", t2-t1, "segundos")
print(w)

#res1=funcoesprovav3.classificaPorW(classe1T,w)
#print(res1)
#---------
#Classe 1 e classe 2B

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2bT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2B foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2C

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2cT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2C foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2D

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2dT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2D foi de", t2-t1, "segundos")
print(w)

#B - parametro 0.01
#velocidade do algorítimo para separar as classes 1 e 2A, 2B, 2C e 2C com paramtro de aprendizagem = 0.01
#Classe 1 e classe 2A

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2aT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("agora com parametro de aprendizagem 0.01")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2A foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2B

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2bT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2B foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2C

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2cT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2C foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2D

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2dT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2D foi de", t2-t1, "segundos")
print(w)

'''
PARA TESTAR O FDA USAR DADOS DO EX2 DA AULA 16
Testar a análise discriminante de Fisher
'''

#Gere um conjunto de dados com 2 classes, 400 padrões (200 padrões por
#classe) e 3 características.

medias=np.array([[-2,6,6],[2,6,6]])
covariancias=np.zeros((2,3,3))
covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
priors=np.array([1/2,1/2])

dados,classes=funcoesprovav3.gerandodadosgaussianos(medias,covariancias,400,priors,plotar=True,seed=0,angulo=[20,120])

#print (np.shape(classes)) #(gabarito)
#print(np.shape(dados)) #(caracteristicasxpadrões)

dadosT = np.transpose(dados) #(padroesxcaracteristicas)print ((np.shape(dadosT)))
#print(np.shape(dadosT))

#dadosT é um vetor do tipo padroesxcaracteristicas. Vamos separa-lo de acordo com as classes. 
classe1=dadosT[classes==0,:]
classe2=dadosT[classes==1,:]
#print (np.shape(classe1))
#print (np.shape(classe2))

#Aplicando a análise discriminante de Fisher (FDA)
#A entrada é uma lista. 
#Portanto, vamos usar a função criarLista primeiro. 

lista = funcoesprovav3.criarListas(classe1,classe2)
#é nessecário dar os dados separados na lista. 

dadosPescados,autoVetores = funcoesprovav3.fda(lista,1)
print (np.shape(dadosPescados))
print ("\n")
print (autoVetores)

#Vamos separar os dados após o FDA
classe1FDA=dadosPescados[classes==0,:]
classe2FDA=dadosPescados[classes==1,:]
print ("\n")
print (np.shape(classe1FDA))
print (np.shape(classe2FDA))

#Agora plotando os gráficos:
    
plt.figure()
plt.title('FDA')
plt.xlabel('padrões')
plt.ylabel('caracteristica otimizada')
plt.plot (classe1FDA[:,0],"bo")
plt.plot (classe2FDA[:,0],"ro")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()
'''

''' Aula 24 - LS, plotar w (perceptron e LS)
#LS - TESTE
#Testar a função acima nos dados simulados do exercício 16
#Da aula 16:
#Gere um conjunto de dados com 2 classes, 400 padrões (200 padrões por classe) e 3 características.

medias=np.array([[-2,6,6],[2,6,6]])
covariancias=np.zeros((2,3,3))
covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
priors=np.array([1/2,1/2])
dados,classes=funcoesprovav4.gerandodadosgaussianos(medias,covariancias,400,priors,plotar=True,seed=0,angulo=[20,120])

#print (np.shape(classes)) #(gabarito)
#print(np.shape(dados)) #(caracteristicasxpadrões)

dadosT = np.transpose(dados) #(padroesxcaracteristicas)
#print ((np.shape(dadosT)))

#----------------------

#o classificador ls recebe os dados concatenados + vetor classes (gabarito)
w2 = funcoesprovav4.ls(dadosT,classes,0)
print (np.shape(w2)) 
print (w2)
print ("\n")
print ("\n")
#Separando as classes usando o classifica perceptron (que usa o w)
#Criando dados com o trem de teste

gabaritoTreino, gabaritoTeste, grupoTreino, grupoTeste = train_test_split(classes, dadosT, train_size = 0.8, random_state = 100)
dTeste = grupoTeste

resultadosTeste = funcoesprovav4.classificaPorW(dTeste,w2)
print ("os dados gerados pelo trem de teste pertencem à classe 1?",resultadosTeste)

#Contabilizando
tamanho = np.shape(resultadosTeste)
grupo1 = np.sum(resultadosTeste) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("\n")
print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")

#Plotando
#Como temos um w (classificador linear), vamos usar a função PlotaSeparaçãoW
#chamando a função. 
funcoesprovav4.plotSeparacaoW(dadosT, w2) 
"""

""" Aula 22 - Perceptron e LDA (FDA - Fisher)
dados = scipy.io.loadmat('dadosex2.mat')
classe1 = dados['Classe1']
classe2a = dados['Classe2a']
classe2b = dados['Classe2b']
classe2c = dados['Classe2c']
classe2d = dados['Classe2d']

print (np.shape(classe1))
print (np.shape(classe2a))
print (np.shape(classe2b))
print (np.shape(classe2c))
print (np.shape(classe2d))

#Ao printar, nota-se que o tamanho dos arquivos é (2x100). 
#Isto é, está no formato caracteristicaxpadrões
#Para colocar no formato padrõesxcaracteristica (trocar linhas e colunas)
#vamos utilizar a transposta. 


classe1T = np.transpose(classe1)
classe2aT = np.transpose(classe2a)
classe2bT = np.transpose(classe2b)
classe2cT = np.transpose(classe2c)
classe2dT = np.transpose(classe2d)

print ("\n")
print (np.shape(classe1T))
print (np.shape(classe2aT))
print (np.shape(classe2bT))
print (np.shape(classe2cT))
print (np.shape(classe2dT))

#A
#velocidade do algorítimo para separar as classes 1 e 2A, 2B, 2C e 2C com paramtro de aprendizagem = 0.05
#Classe 1 e classe 2A

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2aT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2A foi de", t2-t1, "segundos")
print(w)

#res1=funcoesprovav3.classificaPorW(classe1T,w)
#print(res1)
#---------
#Classe 1 e classe 2B

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2bT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2B foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2C

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2cT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2C foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2D

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2dT,rho=0.05,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2D foi de", t2-t1, "segundos")
print(w)

#B - parametro 0.01
#velocidade do algorítimo para separar as classes 1 e 2A, 2B, 2C e 2C com paramtro de aprendizagem = 0.01
#Classe 1 e classe 2A

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2aT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("agora com parametro de aprendizagem 0.01")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2A foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2B

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2bT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2B foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2C

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2cT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2C foi de", t2-t1, "segundos")
print(w)
#---------
#Classe 1 e classe 2D

t1=time.time()
w, iter = funcoesprovav3.perceptron(classe1T,classe2dT,rho=0.01,niter=10000,plot=1)
t2=time.time()
print ("\n")
print ("a quantidade de iterações foi de", iter)
print (" e o tempo para separar as classes 1 e 2D foi de", t2-t1, "segundos")
print(w)

'''
PARA TESTAR O FDA USAR DADOS DO EX2 DA AULA 16
Testar a análise discriminante de Fisher
'''

#Gere um conjunto de dados com 2 classes, 400 padrões (200 padrões por
#classe) e 3 características.

medias=np.array([[-2,6,6],[2,6,6]])
covariancias=np.zeros((2,3,3))
covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
priors=np.array([1/2,1/2])

dados,classes=funcoesprovav3.gerandodadosgaussianos(medias,covariancias,400,priors,plotar=True,seed=0,angulo=[20,120])

#print (np.shape(classes)) #(gabarito)
#print(np.shape(dados)) #(caracteristicasxpadrões)

dadosT = np.transpose(dados) #(padroesxcaracteristicas)print ((np.shape(dadosT)))
#print(np.shape(dadosT))

#dadosT é um vetor do tipo padroesxcaracteristicas. Vamos separa-lo de acordo com as classes. 
classe1=dadosT[classes==0,:]
classe2=dadosT[classes==1,:]
#print (np.shape(classe1))
#print (np.shape(classe2))

#Aplicando a análise discriminante de Fisher (FDA)
#A entrada é uma lista. 
#Portanto, vamos usar a função criarLista primeiro. 

lista = funcoesprovav3.criarListas(classe1,classe2)
#é nessecário dar os dados separados na lista. 

dadosPescados,autoVetores = funcoesprovav3.fda(lista,1)
print (np.shape(dadosPescados))
print ("\n")
print (autoVetores)

#Vamos separar os dados após o FDA
classe1FDA=dadosPescados[classes==0,:]
classe2FDA=dadosPescados[classes==1,:]
print ("\n")
print (np.shape(classe1FDA))
print (np.shape(classe2FDA))

#Agora plotando os gráficos:
    
plt.figure()
plt.title('FDA')
plt.xlabel('padrões')
plt.ylabel('caracteristica otimizada')
plt.plot (classe1FDA[:,0],"bo")
plt.plot (classe2FDA[:,0],"ro")
plt.legend(loc="upper right") #para colocar a legenda
plt.show ()

"""

""" Aula 24 - Testando Perceptron e LS (sozinhos e no k-fold)
import numpy as np
import scipy
import scipy.io
import scipy.stats
from sklearn.model_selection import KFold
import funcoes

#3A
arquivo = scipy.io.loadmat('dadosaula24')
#print (arquivo)
dados = arquivo['dados']
classes = arquivo['classes'].reshape(-1)
#print (np.shape(dados))
#print (np.shape(classes))

dadosT = np.transpose(dados)
#print (np.shape(dadosT))
#print (classes)

classe1=dadosT[classes==1,:]
classe2=dadosT[classes==2,:]
#print (np.shape(classe1))
#print (np.shape(classe2))

#dados (caracteristicasxpadrões) (3x400)
#dadosT (padrõesxcaracteristicas) (400x3)
#classes - vetor gabarito 
#classe1 (200x3) (padrõesxcaracteristicas)
#classe2(200x3) (padrõesxcaracteristicas)
#resultadoTeste - vetor com true (para classe1) e false(para classe2)
#--------------------------------------------------------
#Rodando o Perceptron usando todos os padrões como dados de treinamento
#Depois, calcular nesses mesmos dados, os percentuais de erro (fp e fn). 

w, iter = funcoes.percepoket(classe1, classe2, rho=0.1, niter=10000, plot=1)
resultadosTeste = funcoes.classificaPorW(dadosT, w)

#print (resultadosTeste)
#se for true pertence a primeira classe. Se for false, pertence a segunda. 

#Para contabilizar
tamanho = np.shape(resultadosTeste)
grupo1 = np.sum(resultadosTeste) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")

#Para avaliar o classificador:

vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoes.analiseEstatistica(classes==1,resultadosTeste==True)
print ("a taxa de falsos positivos é", fp)
print ("a taxa de falsos negativos é", fn)
#--------------------------------------------------------
#Rodando o LS usando todos os padrões como dados de treinamento
#Depois, calcular nesses mesmos dados, os percentuais de erro (fp e fn). 

print ("\n")
print ("LS")
#print(classes)
w2 = funcoes.ls(dadosT,classes==1)
resultadosTeste2 = funcoes.classificaPorW(dadosT, w2)

tamanho = np.shape(resultadosTeste2)
grupo1 = np.sum(resultadosTeste2) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")

vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoes.analiseEstatistica(classes==1,resultadosTeste2==True)
print ("a taxa de falsos positivos é", fp)
print ("a taxa de falsos negativos é", fn)

print ("\n")
print ("O Perceptron teve melhor desempenho")
print ("O LS é estatistico, ele tenta prever como o dado irá se comportar, enquando o Perceptron é meio 'força bruta' e funciona bem para casos com poucos padrões. Nesse caso, ele se saiu melhor")
#--------------------------------------------------------
#3B - testar o procedimento usando a validação cruzada k-fold com Perceptron
#ExemploaquiK-fold
scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10) 
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]  
   
        w, iter = funcoes.percepoket(dadosTreino[classesTreino==1,:], dadosTreino[classesTreino==2,:],plot=0)
        
        resultadosTeste = funcoes.classificaPorW(dadosTeste, w)
           
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoes.analiseEstatistica(classesTeste==1,resultadosTeste==True)
       
        fp=fp/np.sum(classesTeste==2)
        fn=fn/np.sum(classesTeste==1)
        scores.append((fp,fn))

#print(np.shape(scores))
print ("\n")
print ("Aplicando k-fold no classificador perceptron")
print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))
#(Exemplo) printar resultado fazendo a média 

#--------------------------------------------------------
#3B - testar o procedimento usando a validação cruzada k-fold com LS


scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10) 
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]  
   
        w = funcoes.ls(dadosTreino, classesTreino==1)
    
        resultadosTeste = funcoes.classificaPorW(dadosTeste, w)
           
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
       
        fp=fp/np.sum(classesTeste==2)
        fn=fn/np.sum(classesTeste==1)
        scores.append((fp,fn))
        #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)
  
#print(np.shape(scores))
print ("\n")
print ("Aplicando k-fold no classificador LS")
print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))
#(Exemplo) printar resultado fazendo a média 

print ("O classificador Perceptron teve um melhor desempenho em relação ao LS.")
print ("Isso é devido ao espalhamento intraclasses da classe 1, que faz com que o LS desloque a reta de classificação na direção dessa classe, gerando mais erros")
"""

""" Aula 24 - k-fold + FDA + Percpetron e LS
import numpy as np
import scipy
import scipy.io
import scipy.stats
from sklearn.model_selection import KFold
import funcoes

#ExemploaquiK-fold-FDA
#Aplicando k-fold + Perceptron + FDA
arquivo = scipy.io.loadmat('dadosaula24')
dados = arquivo['dados']
classes = arquivo['classes'].reshape(-1)
dadosT = np.transpose(dados)
classe1=dadosT[classes==1,:]
classe2=dadosT[classes==2,:]

#dados (caracteristicasxpadrões) (3x400)
#dadosT (padrõesxcaracteristicas) (400x3)
#classes - vetor gabarito (vale 1 ou 2)
#classe1 (200x3) (padrõesxcaracteristicas)
#classe2(200x3) (padrõesxcaracteristicas)
#resultadoTeste - vetor com true (para classe1) e false(para classe2)

scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10) 
    
    for iTreino, iTeste in kf.split(dadosT): 
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]     
        
        #-> uso a função separaclasses nos dados de treino 
        classe1Treino=dadosTreino[classesTreino==1]
        classe2Treino=dadosTreino[classesTreino==2]
        
        #-> uso a função criarListas para criar uma lista de classes como entrada da FDA
        
        listaClasses = funcoes.criarListas(classe1Treino,classe2Treino)
            
        
        
        
        #-> chamo a função FDA
        dadosTreinoFDA,autoVetores = funcoes.fda(listaClasses, 2)
        dadosTesteFDA=dadosTeste.dot(autoVetores)
    
    
    
        #-> 
        w, iter = funcoes.percepoket(dadosTreinoFDA[classesTreino==1], dadosTreinoFDA[classesTreino==2],plot=0)
  
        resultadosTeste = funcoes.classificaPorW(dadosTesteFDA, w)
       
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
 
        fp=fp/np.sum(classesTeste==2)
        fn=fn/np.sum(classesTeste==1)
        scores.append((fp,fn))
        #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)
  
#print(np.shape(scores))
print ("\n")
print ("Aplicando k-fold + FDA no classificador perceptron")
print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))
#(Exemplo) printar resultado fazendo a média 

#Aplicando k-fold + LS + FDA

scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10) 
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]  
        
        
        #-----> FDA <----- A partir de agora, só usar dadosTreinoFDA e dadosTesteFDA, exceto o gabarito

        classe1Treino=dadosTreino[classesTreino==1]
        classe2Treino=dadosTreino[classesTreino==2]
        
        listaClasses = funcoes.criarListas(classe1Treino,classe2Treino)

        dadosTreinoFDA,autoVetores = funcoes.fda(listaClasses, 2)
 
        dadosTesteFDA=dadosTeste.dot(autoVetores)
        
        #-----> FDA <-----
        w = funcoes.ls(dadosTreinoFDA, classesTreino==1)
        
 
        resultadosTeste = funcoes.classificaPorW(dadosTesteFDA, w)
       
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoes.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        
        fp=fp/np.sum(classesTeste==2)
        fn=fn/np.sum(classesTeste==1)
        scores.append((fp,fn))
        #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)
  

#print(np.shape(scores))
print ("\n")
print ("Aplicando k-fold + FDA no classificador LS")
print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))
#(Exemplo) printar resultado fazendo a média 
"""


""" Aula 24 - LS, plotar w (perceptron e LS) (feito antes da p3)
#LS - TESTE
#Testar a função acima nos dados simulados do exercício 16
#Da aula 16:
#Gere um conjunto de dados com 2 classes, 400 padrões (200 padrões por classe) e 3 características.

medias=np.array([[-2,6,6],[2,6,6]])
covariancias=np.zeros((2,3,3))
covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
priors=np.array([1/2,1/2])
dados,classes=funcoesprovav4.gerandodadosgaussianos(medias,covariancias,400,priors,plotar=True,seed=0,angulo=[20,120])

#print (np.shape(classes)) #(gabarito)
#print(np.shape(dados)) #(caracteristicasxpadrões)

dadosT = np.transpose(dados) #(padroesxcaracteristicas)
#print ((np.shape(dadosT)))

#----------------------

#o classificador ls recebe os dados concatenados + vetor classes (gabarito)
w2 = funcoesprovav4.ls(dadosT,classes,0)
print (np.shape(w2)) 
print (w2)
print ("\n")
print ("\n")
#Separando as classes usando o classifica perceptron (que usa o w)
#Criando dados com o trem de teste

gabaritoTreino, gabaritoTeste, grupoTreino, grupoTeste = train_test_split(classes, dadosT, train_size = 0.8, random_state = 100)
dTeste = grupoTeste

resultadosTeste = funcoesprovav4.classificaPorW(dTeste,w2)
print ("os dados gerados pelo trem de teste pertencem à classe 1?",resultadosTeste)

#Contabilizando
tamanho = np.shape(resultadosTeste)
grupo1 = np.sum(resultadosTeste) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("\n")
print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")

#Plotando
#Como temos um w (classificador linear), vamos usar a função PlotaSeparaçãoW
#chamando a função. 
funcoesprovav4.plotSeparacaoW(dadosT, w2) 
'''

Aula 24 - k-fold e FDA

#LS - TESTE
#Testar a função acima nos dados simulados do exercício 16
#Da aula 16:
#Gere um conjunto de dados com 2 classes, 400 padrões (200 padrões por classe) e 3 características.

medias=np.array([[-2,6,6],[2,6,6]])
covariancias=np.zeros((2,3,3))
covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
priors=np.array([1/2,1/2])
dados,classes=funcoesprovav4.gerandodadosgaussianos(medias,covariancias,400,priors,plotar=False,seed=0,angulo=[20,120])

#print (np.shape(classes)) #(gabarito)
#print(np.shape(dados)) #(caracteristicasxpadrões)

dadosT = np.transpose(dados) #(padroesxcaracteristicas)
#print ((np.shape(dadosT)))

#----------------------

#o classificador ls recebe os dados concatenados + vetor classes (gabarito)
w2 = funcoesprovav4.ls(dadosT,classes,0)
print (np.shape(w2)) 
print (w2)
print ("\n")
print ("\n")
#Separando as classes usando o classifica perceptron (que usa o w)
#Criando dados com o trem de teste

gabaritoTreino, gabaritoTeste, grupoTreino, grupoTeste = train_test_split(classes, dadosT, train_size = 0.8, random_state = 100)
dTeste = grupoTeste

resultadosTeste = funcoesprovav4.classificaPerceptron(dTeste,w2)
print ("os dados gerados pelo trem de teste pertencem à classe 1?",resultadosTeste)

#Contabilizando
tamanho = np.shape(resultadosTeste)
grupo1 = np.sum(resultadosTeste) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("\n")
print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")

#Plotando
#Como temos um w (classificador linear), vamos usar a função PlotaSeparaçãoW
#chamando a função. 
funcoesprovav4.plotSeparacaoW(dadosT, w2) 
dados=dadosT[:,0:2]
w=funcoesprovav4.ls(dados, classes, 0)
funcoesprovav4.plotSeparacaoW(dados,w)

dados1=dadosT[:,0:1]
w1=funcoesprovav4.ls(dados1, classes, 0)
print(w1)
funcoesprovav4.plotSeparacaoW(dados1,w1)

x = np.linspace(np.min(dadosT[:,0]), np.max(dadosT[:,0]),2)
y = np.linspace(np.min(dadosT[:,1]), np.max(dadosT[:,1]),2)
print(x)
x , y=np.meshgrid(x, y)
print(x)
z = -(w2[0]*x+w2[1]*y)/w2[2]
#print(np.shape(eixos))
#print("x",x)
    
fig = plt.figure(figsize=(5,5))
plt.title('')
ax = fig.add_subplot(111, projection='3d')
x1 = classe1[:,0]
y1 = classe1[:,1]
z1 = classe1[:,2]

x2 = classe2[:,0]
y2 = classe2[:,1]
z2 = classe2[:,2]

ax.scatter(x1, y1, z1,color='pink',alpha=0.5) #alpha deixa os pontos 50% transparentes(nesse caso)
ax.scatter(x2, y2, z2,color='pink',alpha=0.5)
ax.plot_surface(x,y,z,alpha=0.8)
plt.show()


arquivo = scipy.io.loadmat('Dados_ex3.mat')
dados = arquivo['dados']
classes = arquivo['classes'].reshape(-1)
print (np.shape(dados))
print (np.shape(classes))

#Está no formato (caracteristicasxpadrões) entao vamos fazer a transposta para deixar no formato (padrõesxcaracteristicas)
dadosT = np.transpose(dados)
print (np.shape(dadosT))

#print (classes)

classe1=dadosT[classes==1,:]
classe2=dadosT[classes==2,:]

print (np.shape(classe1))
print (np.shape(classe2))

#dadosT = todos os dados
#classes = gabarito das classes
#classe1 = padrões classificados como "1"
#classe2 = padrões classificados como "2"

#########################################################
#validação k-fold no Perceptron

scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10)   
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        classesTreino, classesTeste = classes[iTreino], classes[iTeste] #classes é o gabarito. Não é obrigatório, mas é preciso usar se o classificador usa o parametro classes/gabarito
       
        w, iter = funcoesprovav5.percepoket(dadosTreino[classesTreino==1,:], dadosTreino[classesTreino==2,:]) 
      
        resultadosTeste = funcoesprovav5.classificaPorW(dadosTeste, w)
        #funcoesprovav5.plotSeparacaoW(dadosTeste, w)
       
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoesprovav5.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        
        fp = 1-especificidade
        fn = 1-sensibilidade
        
    scores.append((fp,fn)) #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)

#print(np.shape(scores))
print("a taxa de FPs e FNs no Perceptron foi de",np.mean(scores,axis=0))

#########################################################
#validação k-fold no LS

scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10)   
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        classesTreino, classesTeste = classes[iTreino], classes[iTeste]
        
        w2 = funcoesprovav5.ls(dadosTreino, classes==1)
        
        resultadosTeste = funcoesprovav5.classificaPorW(dadosTeste, w2)
       
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoesprovav5.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        
        fp = 1-especificidade
        fn = 1-sensibilidade
       
        scores.append((fp,fn)) 
        
fp = 1-especificidade #para obter as taxas
fn = 1-sensibilidade #para obter as taxas

print("a taxa de FPs e FNs no LS foi de",np.mean(scores,axis=0))  #(Exemplo) printar resultado fazendo a média 

print ("O classificador Perceptron teve um melhor desempenho em relação ao LS.")
print ("Isso é devido ao espalhamento intraclasses da classe 1, que faz com que o LS desloque a reta de classificação na direção dessa classe, gerando mais erros")


#########################################################
#Exercício 4 - FDA + k-fold
#Teste o desempenho de uma estratégia baseada em FDA nos mesmos dados, utilizando um procedimento 2-fold com 20 repetições

#dadosT = todos os dados
#classes = gabarito das classes
#classe1 = padrões classificados como "1"
#classe2 = padrões classificados como "2"

lista = funcoesprovav5.criarListas(classe1,classe2)

dadosFDA,autoVetores = funcoesprovav5.fda(lista, 1)

print ("\n")
print("Exercício 4 - FDA + K-fold")
print (np.shape(dadosFDA))

classe1FDA=dadosFDA[classes==1,:]
classe2FDA=dadosFDA[classes==2,:]

print (np.shape(classe1FDA))
print (np.shape(classe2FDA))

#APÓS FDA
#dadosFDA = todos os dados
#classes = gabarito das classes
#classe1FDA = padrões classificados como "1"
#classe2FDA = padrões classificados como "2"
#autoVetores

######

scores=[] 
for i in range (20): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*20)   
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosT[iTreino,:], dadosT[iTeste,:]
        classesTreino, classesTeste = classes[iTreino], classes[iTeste] #classes é o gabarito. Não é obrigatório, mas é preciso usar se o classificador usa o parametro classes/gabarito
       
        classesSeparadas,valClasses = funcoesprovav5.separaClasses(dadosTreino, classesTreino)

        dadosFDA,autoVetores = funcoesprovav5.fda(classesSeparadas,2)
        
        w, iter = funcoesprovav5.percepoket(dadosFDA[classesTreino==1,:], dadosFDA[classesTreino==2,:]) 
      
        dadosTesteFDA=dadosTeste.dot(autoVetores)
              
        resultadosTeste = funcoesprovav5.classificaPorW(dadosTesteFDA, w)
        #funcoesprovav5.plotSeparacaoW(dadosTeste, w)
       
        vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoesprovav5.analiseEstatistica(classesTeste==1, resultadosTeste==True)
        
        fp = 1-especificidade
        fn = 1-sensibilidade
        
    scores.append((fp,fn)) #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)

#print(np.shape(scores))
print("a taxa de FPs e FNs no Perceptron com FDA foi de",np.mean(scores,axis=0))
print("o FDA atrapalhou (dificultou a separação) os dados pq ele deixa os dados apertados")
print ("como a sepação já era linear, ele não era necessário")
print ("o FDA compactou")
print ("aumentando as iterações do perceptron, ele ficaria melhor")      

'''Como utilizar k-fold + FDA
Copiar e colar:
scores=[] 
for i in range (10): #quantidade de vezes para repetir a operação

    kf = KFold(n_splits=2, shuffle=True, random_state=i*10)   
    
    for iTreino, iTeste in kf.split(dadosT):    
        
        dadosTreino, dadosTeste = dadosTotais[iTreino,:], dadosTotais[iTeste,:]
        classesTreino, classesTeste = classes[iTreino], classes[iTeste] #classes é o gabarito. Não é obrigatório, mas é preciso usar se o classificador usa o parametro classes/gabarito
       
       
       -> uso a função separaclasses nos dados de treino 
       -> chamo a função FDA
       
       
       --->>> [inserir o classificador aqui com os parâmetros]<-----
        -> o parâmetro "classe1, classe2" será dadosTreino[classesTreino==1,:], dadosTreino[classesTreino==2,:], que é o que o módulo classificou como sendo as classes. 
        -> ==1 ==2 ou ==0 ==1 ... depende do valor dado nas classes
        -> Lembrar de usar dadosTreino em vez dos dados Totais (se no LS).
        -> os dados totais NAO EXISTEM dentro do k-fold.
              
       ----->[inserir o classifica por w, no caso do perceptron e LS]<---------
       -> o parametro será dadosTeste, que é a fatia que o k-fold elegeu para teste
       
       
       ->agora, multilpicar os dadosTeste pelos autovetores do FDA
       ->dadosTesteFDA=dadosTeste.dot(autoVetores)
       
       ------>[inserir a análise estatistica/função que vê sensibilidade, vp, etc. ]<---------
       -> usar classesTeste==1, resultadosTeste==True
        
        scores.append((fp,fn)) #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)
#print(np.shape(scores))

fp = 1-especificidade #para obter as taxas
fn = 1-sensibilidade #para obter as taxas

print("a taxa de FPs e FNs foi de",np.mean(scores,axis=0))  #(Exemplo) printar resultado fazendo a média 
"""

""" Extra: plotando curvas de nível
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import funcoesAntigo
import funcoes

def valoresPossiveis(lista):
    #essa função recebe um vetor e retorna todos os valores desse vetor, ordenados e sem repetição
    #foi usada no classificador bayes
    possiveis=[]
    while(np.shape(lista)[0]>0):
        min=np.min(lista)
        possiveis.append(min)
        lista=lista[lista!=min]
    return possiveis

def plotGausCurvadeNivel(dados,classes):
   
    val= valoresPossiveis(classes)
    fig=plt.figure(figsize=(8,8))
    c=0
    for i in val:
        X=dados[classes==i,:]

        # Extract x and y
        x = X[:, 0]
        y = X[:, 1]
        # Define the borders
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10
        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        mi=xmin
        if (ymin<mi):
            mi=ymin
        ma=xmax
        if(ymax>ma):
            ma=ymax

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        cor=['blue', 'red', 'green', 'black',  'brown']
       
        ax = fig.gca()
        cset = ax.contour(xx, yy, f, colors='k')
        for j in range(len(cset.allsegs)):
            for ii, seg in enumerate(cset.allsegs[j]):
                plt.plot(seg[:,0], seg[:,1], '-',color=cor[c])
        plt.xlim((mi,ma))
        plt.ylim((mi,ma))
        #plt.legend()
        c+=1

'''
medias=np.array([[-2,6,6],[2,6,6]])
covariancias=np.zeros((2,3,3))
covariancias[0,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
covariancias[1,:,:]=np.array([[0.3,1,1],[1,9,1],[1,1,9]])
priors=np.array([1/2,1/2])
'''

m1=(0,0)
m2=(0,0)
#m3=(0,0)
#m4=(2,-2)
cov1=[[5,10],[10,1]] #termo1,1 - termo 1,2 - termo 2,1 - termo2,2 respectivamente
cov2=[[10,1],[1,5]]
#cov3=[[0.5,0],[0,0.1]]
#cov4=[[1,0.5],[0.5,1]]
medias=funcoes.criarListas(m1,m2)
covariancias=funcoes.criarListas(cov1,cov2)
priors=np.array([1/2,1/2])
dados,classes=funcoes.gerandodadosgaussianos(medias,covariancias,4000,priors,plotar=False,seed=0,angulo=[20,120])
dados=dados.T

plotGaus(dados[classes==0],classes[classes==0])

"""

""" P3 - refeita 
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
import funcoesp3

arquivo = scipy.io.loadmat('P3.mat')
pensOK = arquivo['PensOK']
pensBanido = arquivo['PensBanido']
#print (np.shape(pensOK))
#print (np.shape(pensBanido))

#Para criar um vetor de classes

dadosOKeBan,classes = funcoesp3.concatenarClasses(pensOK,pensBanido)
#print ("\n")
#print (np.shape(dadosOKeBan))
#print (np.shape(classes))
#print (classes)

classeOK = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],)
classeBan = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,])
#print (np.shape(classeOK))
#print (np.shape(classeBan))
#print (classeOK)
#print (classeBan)

#pensOK (108x1000) (padrõesxcaracteristicas)
#pensBanido (87x1000) (padrõesxcaracteristicas)
#dadosOKeBan (195x1000) (padrõesxcaracteristicas)
#classes (195)
#classeOK (108)
#classeBan(87)

#####################################
#Criando o classificador:

def scatter2(dados):
    '''Usado na FDA'''
    # Dados de treinamento (divididos em três classes)
    #X = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14]])
    #y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    #Calcula duas matrizes dentro de FDA (Fisher)
    
    X=0
    y=0
    for i in range(np.shape(dados)[0]):
        if (i==0):
            X=dados[i]
            y=np.zeros((np.shape(dados[i])[0]))
        else:
            X=np.concatenate((X,dados[i]),axis=0)
            y=np.concatenate((y,i*np.ones((np.shape(dados[i])[0]))))
    # Cálculo das médias das classes
    val=valoresPossiveis(y)
    mean_vectors = []
    for cl in val:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    
    # Cálculo da matriz de covariância dentro das classes
    S_W = np.zeros((np.shape(X)[1], np.shape(X)[1]))
    for cl,mv in zip(val, mean_vectors):
        class_sc_mat = np.cov(X[y==cl].T) # calcula a matriz de covariância da classe
        S_W += class_sc_mat
    
    # Cálculo da matriz de covariância entre as classes
    mean_overall = np.mean(X, axis=0)
    S_B = np.zeros((np.shape(X)[1], np.shape(X)[1]))
    for i,mean_vec in enumerate(mean_vectors):
        n = np.shape(X[y==i+1,:])[0]
        mean_vec = mean_vec.reshape(np.shape(X)[1],1) # reshaping
        mean_overall = mean_overall.reshape(np.shape(X)[1],1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    
    # matriz de fisher
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    
    # ordenando os autovetores pela ordem dos autovalores
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    
    return S_W,S_B

def autoordenar(eigvec, eigval):
    '''Usado na FDA'''
    # eigvalord = sorted(eigval, reverse=True)
    idx = np.argsort(eigval)
    eigvec = eigvec.T
    eigvecord = eigvec[idx]
    eigvalord = eigval[idx]
    eigvecord = eigvecord[-1::-1]
    eigvalord = eigvalord[-1::-1]
    return eigvecord.T, eigvalord

def valoresPossiveis(lista):
    #essa função recebe um vetor e retorna todos os valores desse vetor, ordenados e sem repetição
    #foi usada no classificador bayes
    possiveis=[]
    while(np.shape(lista)[0]>0):
        min=np.min(lista)
        possiveis.append(min)
        lista=lista[lista!=min]
    return possiveis

def classificadorp3 (listaClasses, n, rho=0.1, niter=10000, plot=0):
    
    '''
    Entradas: 
        ListaClasses: lista de classes do tipo (padrõesxcaracteristicas) *usar a função criar listas para fazer a entrada
        n: nº de caracteristicas desejadas. Nesse caso, 1. 
        rho: Parâmetro de aprendizado. Pequeno: demora. Grande: pode não ser preciso.
        niter:número máximo de passos. Valor em que o sistema para. 
          
    Saídas: 
    dadosPescados (piadinha com fisher) = os dados nos novos eixos
    autoVetores = usados para colocar uma amostra de teste no mesmo "idioma/angulo"
    w: o hiperplano que separa as classes - vetor com os pesos do classificador linear
    os dois primeiros valores do vetor sao o peso. Se eles forem menores (em módulo) que o terceiro valor,
    então os pontos pertencem ao grupo 1. Se eles forem maiores (em módulo), os pontos pertencem ao grupo 2
    iter: quantos passos foram necessários para calcular (w)
    '''

    Sw, Sb = scatter2(listaClasses)
    eigval, eigvec = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

    eigvec, eigval = autoordenar(eigvec, eigval)

    C = len(listaClasses)  # Number of classes.
    autoVetores = np.abs(eigvec[:, 0:n])

    X = listaClasses[0]
    y=np.zeros((np.shape(listaClasses[0])[0]))

    for n in range(1, C):
        X = np.concatenate((X, listaClasses[n]), axis=0)
        y=np.concatenate((y,n*np.ones((np.shape(listaClasses[n])[0]))))
    dadosPescados = X.dot(autoVetores)
    
    classe1FDA=dadosPescados[y==0,:]
    classe2FDA=dadosPescados[y==1,:]

    classe1FDA = np.transpose(classe1FDA)
    classe2FDA = np.transpose(classe2FDA)
    J = lambda p, w, dados, Y: sum(p[Y] * w.dot(dados[:, Y]))
    L, N1 = classe1FDA.shape
    L2, N2 = classe2FDA.shape
    iter = niter

    c = np.ones([1, N1 + N2])
    c[0, N1:] = -1

    resultado = "convergiu em"
    if L != L2:
        print('As classes precisam ter o mesmo numero de caracteristicas!')
        return
    dados = np.ones([L + 1, N1 + N2])
    dados[:-1, :N1] = classe1FDA
    dados[:-1, N1:] = classe2FDA

    w = np.random.randn(L + 1)
    p = np.sign(w.dot(dados))

    inutil, Y = np.where(c != p)
    e = []
    erroAntes = np.zeros(L + 1)
    erroSum = np.zeros(L + 1)
    Kp = rho
    Ki = 0
    Kd = 0
    for n in range(0, niter + 1):
        erro = sum((p[Y] * dados[:, Y]).T)
        P = Kp * erro
        I = Ki * (erroSum + erro)
        D = Kd * (erro - erroAntes) / 0.4
        PD = P + I + D
        wp = w - PD
        erroAntes = erro
        e.append(J(p, wp, dados, Y))

        if J(p, wp, dados, Y) < J(p, w, dados, Y):
            w = wp

        p = np.sign(w.dot(dados))
        inutil, Y = np.where(c != p)
        if np.where(c != p)[0].shape[0] == 0:
            iter = n + 1
            break
    if n == niter:
        resultado = "não converge com"
    if plot == 1:
        # para o erro -> PARA HABILITAR: tirar as '#'
        # plt.figure() Gráfico do erro
        # plt.plot(e)
        # plt.title('Erro do perceptron pocket')
        # plt.show()

        xmin = np.min([np.min(classe1FDA[0, :]), np.min(classe2FDA[0, :])])
        xmax = np.max([np.max(classe1FDA[0, :]), np.max(classe2FDA[0, :])])
        # descobrir o mínimo e o máximo da caracteristica 'x' (horizontal no gráfico)

        ymin = np.min([np.min(classe1FDA[1, :]), np.min(classe2FDA[1, :])])
        ymax = np.max([np.max(classe1FDA[1, :]), np.max(classe2FDA[1, :])])
        # descobrir o mínimo e o máximo da caracteristica 'y' (vertical no gráfico)

        eixoX = np.linspace(xmin, xmax, 1000)  # vetor em x q vai do minimo ao maximo com 1000 pontos
        eixoY = (-w[2] - w[0] * eixoX) / w[1]  # calculo do eixo "y", com base no x e no w obtido
        eixo = (eixoY > ymin) * (eixoY < ymax)  # calcula os limites dos eixos
        eixoX = eixoX[eixo]  # impoem os limites dos eixos
        eixoY = eixoY[eixo]

        plt.figure()
        plt.title('Perceptron - ' + resultado + ': ' + str(iter) + ' iterações, p = ' + str(rho))  # transformar o iter em string
        plt.plot(classe1FDA[0, :], classe1FDA[1, :], ".b")
        plt.plot(classe2FDA[0, :], classe2FDA[1, :], ".r")
        plt.plot(eixoX, eixoY, "b")

        plt.show()

    return dadosPescados,autoVetores, w, iter

def classificaPorW(dTeste,w): 
    
    #dTeste = matriz MxL (amostras x caracteriscas) que será classificada
    #w = vetor de classificador linear (saida do perceptron)
    
    #resultadosTeste = um vetor Mx1 (numero de amostras) com a suposta classificação de cada amostra do dTeste 
    #se for true pertence à primeira classe
    #se for false, é da segunda classe
    #Para plotar, usar plotSeparacaoW
    '''
    Para contabilizar as amostras que pertecem ao grupo true e false, no caso de um trem de teste:
    tamanho = np.shape(resultadosTeste)
    grupo1 = np.sum(resultadosTeste) #ele só soma os trues
    grupo2 = (tamanho-grupo1)
    print ("\n")
    print ("Das", tamanho, "amostras", grupo1, "pertencem ao grupo 1 e", grupo2, "pertencem ao grupo 2")
      
    '''
    
    cut=0
    if (type(dTeste)==list):
        dTeste=[dTeste,dTeste]
        cut=1
    t=np.shape(w)[0]
    dTeste=dTeste*w[:t-1]
    dTeste=np.sum(dTeste,axis=1)
    resultadosTeste=(dTeste+w[t-1])>0

    if(cut==1):
        resultadosTeste=resultadosTeste[0]
    
    return resultadosTeste


listaClasses = funcoesp3.criarListas(pensOK,pensBanido)

dadosPescados,autoVetores, w, iter = classificadorp3(listaClasses,1)

dados,gabarito=funcoesp3.concatenarClasses(pensOK,pensBanido)
resultadosTeste = classificaPorW(dados, w)
#print (resultadosTeste)

tamanho = np.shape(resultadosTeste)
grupo1 = np.sum(resultadosTeste) #ele só soma os trues
grupo2 = (tamanho-grupo1)
print ("\n")
print ("1A")
print ("Segundo o classificador p3, das", tamanho, "amostras banidas,", grupo1, "pertencem ao grupo de pensamos ok e", grupo2, "pertencem ao grupo de pensamentos banidos")
 
vp, vn, fp, fn, sensibilidade,especificidade,precisao,prevalencia,acuracia = funcoesp3.analiseEstatistica (gabarito==0,resultadosTeste)

print ("portanto, a sensibilidade e a específidade foram, respectivamente:",sensibilidade,especificidade)

print ("\n")
print ("1B")

print ("Como temos uma quantidade enorme de caracteristicas, o classificador criado para esse modelo utilizou a FDA para selecionar a caracteristica mais relevante")
print ("isto é, adaptou-se o classificador com o conceito de FDA.")
print ("Isso foi relevante porque a estratégia de Fisher transforma o espaço de caracteristicas tomando as classes como guia, quantificando o espalhamento intraclasses e reduzindo a dimensão dos dados")

print ("Classificadores bayesianos e de distância mínima foram descartados, devido à quantidade de dados e a distância da superfície de decisão")
print ("Por ser robusto ao espalhamento interclasses, escolheu-se partir do classificador Perceptron, com algumas alterações, conforme mostrado na função criada classificadorp3")    
print ("para utilizar a saída w (o vetor de classificador linear adaptado), usou-se a função classificaporw, conforme mostrado e explicado nos comentários da função")
       
#############################
#2


scoresTeste=[] 
scoresTreino=[] 
for i in range (10): #quantidade de vezes para repetir a operação
    print("i=",i)
    kf = KFold(n_splits=2, shuffle=True, random_state=i*10)   
    
    for iTreino, iTeste in kf.split(dadosOKeBan):    
        
        dadosTreino, dadosTeste = dadosOKeBan[iTreino,:], dadosOKeBan[iTeste,:]
        classesTreino, classesTeste = classes[iTreino], classes[iTeste] #classes é o gabarito. Não é obrigatório, mas é preciso usar se o classificador usa o parametro classes/gabarito

        
        dadosPescados,autoVetores, w, iter = classificadorp3([dadosTreino[classesTreino==0],dadosTreino[classesTreino==1]],1)
        
        resultadosTreino = classificaPorW(dadosTreino, w) #dadosTeste, que é a fatia que o k-fold elegeu para teste
        
        vp, vn, fp, fn, sensibilidade, especificidade, precisao, prevalencia, acuracia1 = funcoesp3.analiseEstatistica(classesTreino==0, resultadosTreino==True)
       
        scoresTreino.append((sensibilidade, especificidade)) 
       
        resultadosTeste = classificaPorW(dadosTeste, w) #dadosTeste, que é a fatia que o k-fold elegeu para teste
        
        vp, vn, fp, fn, sensibilidade, especificidade, precisao, prevalencia, acuracia2 = funcoesp3.analiseEstatistica(classesTeste==0, resultadosTeste==True)
       
        scoresTeste.append((sensibilidade, especificidade)) 

        #scores.append((fp,fn)) #saída que estou quantificando (por exemplo, fp, fn, vp, etc.)

#print ("acurácia nos dados de treinamento", acuracia1)
#print ("acurácia nos dados de teste", acuracia2)
#print ("sobreajustado ou subajustado?", acuracia)

print("Treino",np.mean(scoresTreino,axis=0))
print("teste",np.mean(scoresTeste,axis=0))
"""

