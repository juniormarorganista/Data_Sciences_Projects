import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class Modelo:
    def __init__(self, model_type="SVC"):
        """
        Inicializa o modelo com o tipo especificado.
        
        Parâmetros:
        - model_type (str): Tipo de modelo a ser usado. Opções: "SVC", "LinearRegression", "DecisionTree", "RandomForest".
        """
        self.df = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.set_model(model_type)

    def set_model(self, model_type):
        """
        Define o modelo com base no tipo fornecido.
        
        Parâmetros:
        - model_type (str): Tipo de modelo. Pode ser "SVC", "LinearRegression", "DecisionTree" ou "RandomForest".
        """
        if model_type == "SVC":
            self.model = SVC()
        elif model_type == "LinearRegression":
            self.model = LinearRegression()
        elif model_type == "DecisionTree":
            self.model = DecisionTreeClassifier()
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier()
        else:
            raise ValueError("Modelo não suportado. Escolha 'SVC', 'LinearRegression', 'DecisionTree' ou 'RandomForest'")
    
    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
        self.df.dropna(inplace=True)
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def Treinamento(self):
        self.model.fit(self.X_train, self.y_train)

    def Teste(self):
        predictions = self.model.predict(self.X_test)
        if hasattr(self.model, "predict_proba") or hasattr(self.model, "decision_function"):
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions, average='weighted')
            recall = recall_score(self.y_test, predictions, average='weighted')
            print(f"Acurácia: {accuracy}")
            print(f"Precisão: {precision}")
            print(f"Revocação: {recall}")
        else:
            print("Este modelo não é classificador, então não é possível calcular acurácia, precisão e recall.")

    def Train(self, path):
        self.CarregarDataset(path)
        self.TratamentoDeDados()
        self.Treinamento()
        self.Teste()

