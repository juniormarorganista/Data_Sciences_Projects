
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, classification_report, 
                             ConfusionMatrixDisplay, confusion_matrix)

from sklearn.datasets import (load_iris, load_wine, load_breast_cancer, 
                              load_digits, load_diabetes,
                              make_classification, make_blobs, 
                              make_moons, make_circles, make_regression)

from yellowbrick.classifier import ConfusionMatrix

class Coursera:

    def __init__(self, model_type="SVC", data_set_net="iris", **kwargs):
        """
        Inicializa o modelo de acordo com o tipo especificado.
        Parametros:
        - model_type (str): Tipo de modelo a ser usado. 
        Opcoes: 
            - "SVC"
            - "LinearRegression"
            - "DecisionTree"
            - "RandomForest"
        """
        self.df = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model_type = model_type

        self.SetModel(**kwargs)

        self.data_set_net = data_set_net

    def SetModel(self, **kwargs):
        """
        Define o modelo com base no tipo fornecido.
        Parametros:
        - model_type (str): Tipo de modelo. Tipo de modelo a ser usado. 
        Opcoes: 
            - "SVC"
            - "LinearRegression"
            - "DecisionTree"
            - "RandomForest"
        """

        if self.model_type == "SVM":
            self.model = SVC(**kwargs)
        elif self.model_type == "LinearRegression":
            self.model = LinearRegression(**kwargs)
        elif self.model_type == "KNN":
            self.model = KNeighborsClassifier(**kwargs)
        elif self.model_type == "MPLC":
            self.model = MLPClassifier(**kwargs)
        elif self.model_type == "GaussianNB":
            self.model = GaussianNB(**kwargs)
        elif self.model_type == "DecisionTree":
            self.model = DecisionTreeClassifier(**kwargs)
        elif self.model_type == "RandomForest":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("Modelo nao suportado. Escolha entre: 'SVC', 'LinearRegression', 'MPLC', 'GaussianNB', 'KNN', 'DecisionTree' ou 'RandomForest'")
    
    def LoadData(self, path=None):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV ou 
        do dataset Iris embutido no sklearn.
        Parametros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        """
        if path is not None:
            # Carrega o dataset do arquivo CSV
            self.df = pd.read_csv(path)
        else:
            
            # Carrega o dataset
            if self.data_set_net == "iris":
                Iris = load_iris()
                self.df = pd.DataFrame(data=Iris.data, columns=Iris.feature_names)
                self.df['Species'] = Iris.target
            elif self.data_set_net == "wine":
                Wine = load_wine()
                self.df = pd.DataFrame(data=Wine.data, columns=Wine.feature_names)
                self.df['Species'] = Wine.target
            elif self.data_set_net == "cancer":
                Breast_Cancer = load_breast_cancer()
                self.df = pd.DataFrame(data=Breast_Cancer.data, columns=Breast_Cancer.feature_names)
                self.df['Species'] = Breast_Cancer.target
            elif self.data_set_net == "digits":
                Digits = load_digits()
                self.df = pd.DataFrame(data=Digits.data, columns=Digits.feature_names)
                self.df['Species'] = Digits.target
            elif self.data_set_net == "diabetes":
                Diabetes = load_diabetes()
                self.df = pd.DataFrame(data=Diabetes.data, columns=Diabetes.feature_names)
                self.df['Species'] = Diabetes.target
            
            # self.df.info()
            # print(self.df.head(5))

    def TratamentoDeDados(self):
        """
        Realiza o pre-processamento dos dados carregados.

        Sugestoes para o tratamento dos dados:
            * Visualize as primeiras linhas e entenda a estrutura.
            * Verifique a presenca de valores ausentes e faca o tratamento adequado.
            * Considere remover colunas ou linhas que nao sao uteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore graficos e visualizacoes para obter insights sobre a distribuicao dos dados.
            * Certifique-se de que os dados estao limpos e prontos para serem usados no treinamento do modelo.
        """

        # print(f"Data set considerado: {self.df.head(7)}")
        # print(15*'#---')
        # print(self.df.info())
        print(15*'#---')
        
        # Removendo valores ausentes, se existirem
        self.df.dropna(inplace=True)
        
        # Separando features (X) e target (y)
        self.X = self.df.iloc[:, :-1]  # Todas as colunas exceto a ultima
        self.y = self.df.iloc[:, -1]   # Apenas a ultima coluna (target)

        scaler_x = StandardScaler()
        self.X = scaler_x.fit_transform(self.X)

        self.label_encoder_flor = LabelEncoder()
        self.y = self.label_encoder_flor.fit_transform(self.y)

    def Training(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a funcao `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar.
            * Experimente tecnicas de validacao cruzada (cross-validation) para melhorar a acuracia final.
        
        Nota: Esta funcao deve ser ajustada conforme o modelo escolhido.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
        # Treinando o modelo
        self.model.fit(self.X_train, self.y_train)

    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta funcao deve ser implementada para testar o modelo e calcular metricas de avaliacao relevantes, 
        como acuracia, precisao, ou outras metricas apropriadas ao tipo de problema.
        """
        
        self.predictions = self.model.predict(self.X_test)

        # print(classification_report(self.y_test, self.predictions))
        if self.model_type in ["SVM", "DecisionTree", "KNN", "RandomForest", "MPLC", "GaussianNB"]:
            print("#---   "+f"{self.model_type}"+"   ...   "+f"{len(self.model_type)}")
            print(15*'#---')
            accuracy = accuracy_score(self.y_test, self.predictions)
            precision = precision_score(self.y_test, self.predictions, average='weighted')
            recall = recall_score(self.y_test, self.predictions, average='weighted')
            print(f"Acuracia: {accuracy:.6f}")
            print(f"Precisao: {precision:.6f}")
            print(f"Revocacao: {recall:.6f}")
        elif isinstance(self.model, LinearRegression):
            print("Este modelo e de regressao; metricas de classificacao nao sao aplicaveis.")
        
        # cm_display = ConfusionMatrixDisplay.from_predictions(self.y_test, self.predictions, display_labels=self.label_encoder_flor.classes_)
        # # cm_display.plot(cmap=plt.cm.Blues)
        # plt.title("Matriz de Confusao")
        # plt.show()

    def Train(self, path=None):
        """
        Funcao principal para o fluxo de treinamento do modelo.

        Este metodo encapsula as etapas de carregamento de dados, pre-processamento e treinamento do modelo.
        Sua tarefa e garantir que os metodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Parametros:
        - path (str): Caminho opcional para o arquivo CSV com o dataset.
        
        Notas:
            * O dataset padrao e o Iris embutido no sklearn, mas o caminho pode ser ajustado para um arquivo CSV.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.LoadData(path)  # Carrega o dataset especificado ou o dataset Iris embutido
        self.TratamentoDeDados()  # Tratamento de dados opcional
        self.Training()  # Executa o treinamento do modelo
        self.Teste()  # Executa a avaliacao do modelo


# pathdata = "Data/credit_data.csv"
pathdata = None

dat = ["iris", "wine", "cancer", "digits"]

for i in dat:
    print(50*"***")
    print(i)
    print(50*"***")

    # SVM com kernel polinomial e C maior
    modelo_svc = Coursera(model_type="SVM", data_set_net=i, C=10, kernel='poly', degree=3)
    modelo_svc.Train(path=pathdata)

    # Random Forest com 200 arvores e profundidade maxima de 10
    modelo_rf = Coursera(model_type="RandomForest", data_set_net=i, n_estimators=200, max_depth=10)
    modelo_rf.Train(path=pathdata)

    # Decision Tree com profundidade maxima de 5 e criterio de entropia
    modelo_dt = Coursera(model_type="DecisionTree", data_set_net=i, max_depth=5, criterion='entropy')
    modelo_dt.Train(path=pathdata)

    # KNN com kernel polinomial e C maior
    modelo_knn = Coursera(model_type="KNN", data_set_net=i, n_neighbors=5, metric='minkowski', p = 2)
    modelo_knn.Train(path=pathdata)

    # MPLC com kernel polinomial e C maior
    modelo_mplc = Coursera(model_type="MPLC", data_set_net=i, max_iter = 2000, activation = 'relu', batch_size = 56, solver = 'adam')
    modelo_mplc.Train(path=pathdata)

    # SVM com kernel polinomial e C maior
    modelo_gnb = Coursera(model_type="GaussianNB", data_set_net=i)
    modelo_gnb.Train(path=pathdata)

    # Linear Regression com intercepto desativado
    modelo_lr = Coursera(model_type="LinearRegression", data_set_net=i, fit_intercept=False)
    modelo_lr.Train(path=pathdata)