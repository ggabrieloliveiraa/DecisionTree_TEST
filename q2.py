import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
base = pd.read_csv('restaurantev2.csv', encoding='latin-1', sep=';') # ler arquivo csv
base = base.drop('Exemplo', axis=1)
#print(base)

np.unique(base['conc'], return_counts=True)
sns.countplot(x = base['conc'])
X_prev = base.iloc[:, 0:10].values
X_prev_label = base.iloc[:, 0:10]
#print(X_prev_label)
y_classe = base.iloc[:, 10].values



# tratamento de dados categoricos 
label_encoder = LabelEncoder()
label_encoder_Alternativo = LabelEncoder()
label_encoder_Bar = LabelEncoder()
label_encoder_SexSab = LabelEncoder()
label_encoder_fome = LabelEncoder()
label_encoder_chuva = LabelEncoder()
label_encoder_Res = LabelEncoder()

label_encoder_clientes = LabelEncoder()
label_encoder_preco = LabelEncoder()
label_encoder_tempo = LabelEncoder()

#print(X_prev[:,0])
X_prev[:,0] = label_encoder_Alternativo.fit_transform(X_prev[:,0])
X_prev[:,1] = label_encoder_Bar.fit_transform(X_prev[:,1])
X_prev[:,2] = label_encoder_SexSab.fit_transform(X_prev[:,2])
X_prev[:,3] = label_encoder_fome.fit_transform(X_prev[:,3])
#pula clientes
#X_prev[:,4] = label_encoder_clientes.fit_transform(X_prev[:,4])
#pula preco
#X_prev[:,5] = label_encoder_preco.fit_transform(X_prev[:,5])

X_prev[:,6] = label_encoder_chuva.fit_transform(X_prev[:,6])
X_prev[:,7] = label_encoder_Res.fit_transform(X_prev[:,7])
#pula tempo

X_prev[:,9] = label_encoder_tempo.fit_transform(X_prev[:,9])


#X_prev[:,6] = label_encoder_Alternativo.fit_transform(X_prev[:,6])
#X_prev[:,7] = label_encoder_Bar.fit_transform(X_prev[:,7])
#X_prev[:,9] = label_encoder_SexSab.fit_transform(X_prev[:,9])


#binarizando atributos não ordinais(clientes) - OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
len(np.unique(base['Cliente']))
indices = [4, 5, 8] 
onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), indices)], remainder='passthrough')
X_prev= onehotencoder.fit_transform(X_prev)

#método de amostragem holdout
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 23)

#algoritmo decision tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste) #testando modelo

print(confusion_matrix(y_teste, previsoes))
cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
print(cm.score(X_teste, y_teste))

print(classification_report(y_teste, previsoes))
from sklearn import tree

previsores = ['Frances', 'Hamburguer', 'Italiano', 'Tailandes', 'Alternativo', 'Bar', 'SextaSabado', 'Fome', 'Cliente', 'R', 'RR', 'RRR', 'Chuva','Res','Tipo', '0-10', '10-30', '30-60', '>60']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
tree.plot_tree(modelo, feature_names=previsores, class_names = ['Nao', 'Sim'], filled=True)
 #tree.plot_tree(Y)
plt.show()


