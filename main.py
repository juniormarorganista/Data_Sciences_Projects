import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

class Modelo:
    def __init__(self, model_type="SVC", **kwargs):
        """
        Inicializa o modelo de acordo com o tipo especificado.
        Parametros:
        - model_type (str): Tipo de modelo a ser usado. Opcoes: "SVC", "LinearRegression", "DecisionTree", "RandomForest".
        """
        self.df = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model_type = model_type
        self.set_model(**kwargs)

    def set_model(self, **kwargs):
        """
        Define o modelo com base no tipo fornecido.
        Parametros:
        - model_type (str): Tipo de modelo. Pode ser "SVC", 
        "LinearRegression", "DecisionTree" ou "RandomForest".
        """
        if self.model_type == "SVC":
            self.model = SVC(**kwargs)
        elif self.model_type == "LinearRegression":
            self.model = LinearRegression(**kwargs)
        elif self.model_type == "DecisionTree":
            self.model = DecisionTreeClassifier(**kwargs)
        elif self.model_type == "RandomForest":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("Modelo nao suportado. Escolha 'SVC', 'LinearRegression', 'DecisionTree' ou 'RandomForest'")
    
    def CarregarDataset(self, path=None):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV ou 
        do dataset Iris embutido no sklearn.
        Parametros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        O dataset e carregado com as seguintes colunas: SepalLengthCm, 
        SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        if path:
            # Carrega o dataset do arquivo CSV
            names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
            self.df = pd.read_csv(path, names=names)
        else:
            # Carrega o dataset Iris diretamente do sklearn
            iris = load_iris()
            self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            self.df['Species'] = iris.target
    
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

        print(f"Data set considerado: {self.df.head(7)}")
        print(30*'-*-')
        print(self.df.info())
        print(30*'-*-')
        
        # Removendo valores ausentes, se existirem
        self.df.dropna(inplace=True)
        
        # Separando features (X) e target (y)
        self.X = self.df.iloc[:, :-1]  # Todas as colunas exceto a ultima
        self.y = self.df.iloc[:, -1]   # Apenas a ultima coluna (target)

        scaler_x = StandardScaler()
        self.X = scaler_x.fit_transform(self.X)

        self.label_encoder_flor = LabelEncoder()
        self.y = self.label_encoder_flor.fit_transform(self.y)

    def Treinamento(self):
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

        print(classification_report(self.y_test, self.predictions))

        if self.model_type in ["SVC", "DecisionTree", "RandomForest"]:
            accuracy = accuracy_score(self.y_test, self.predictions)
            precision = precision_score(self.y_test, self.predictions, average='weighted')
            recall = recall_score(self.y_test, self.predictions, average='weighted')
            print(f"Acuracia: {accuracy}")
            print(f"Precisao: {precision}")
            print(f"Revocacao: {recall}")
        elif isinstance(self.model, LinearRegression):
            print("Este modelo e de regressao; metricas de classificacao nao sao aplicaveis.")
        
        cm_display = ConfusionMatrixDisplay.from_predictions(self.y_test, self.predictions, display_labels=self.label_encoder_flor.classes_)
        # cm_display.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusao")
        plt.show()

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
        self.CarregarDataset(path)  # Carrega o dataset especificado ou o dataset Iris embutido
        self.TratamentoDeDados()  # Tratamento de dados opcional
        self.Treinamento()  # Executa o treinamento do modelo
        self.Teste()  # Executa a avaliacao do modelo

# SVM com kernel polinomial e C maior
# modelo_svc = Modelo(model_type="SVC", C=10, kernel='poly', degree=3)
# modelo_svc.Train(path="Data/iris.data")

# # Random Forest com 200 arvores e profundidade maxima de 10
modelo_rf = Modelo(model_type="RandomForest", n_estimators=200, max_depth=10)
modelo_rf.Train(path="Data/iris.data")

# # Decision Tree com profundidade maxima de 5 e criterio de entropia
# modelo_dt = Modelo(model_type="DecisionTree", max_depth=5, criterion='entropy')
# modelo_dt.Train(path="Data/iris.data")

# # Linear Regression com intercepto desativado
# modelo_lr = Modelo(model_type="LinearRegression", fit_intercept=False)
# modelo_lr.Train(path="Data/iris.data")