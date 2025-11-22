# ======================================
# PROJETO DE CLASSIFICAÇÃO - MUSHROOM
# UCI MACHINE LEARNING REPOSITORY
# https://archive.ics.uci.edu/dataset/73/mushroom
# ======================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ======================================
# 1. Carregar Base de Dados
# ======================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

columns = [
    'class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
    'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
    'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
    'stalk_surface_below_ring', 'stalk_color_above_ring',
    'stalk_color_below_ring', 'veil_type', 'veil_color',
    'ring_number', 'ring_type', 'spore_print_color', 'population',
    'habitat'
]

df = pd.read_csv(url, header=None, names=columns)

print("Prévia do dataset:")
print(df.head())

# ======================================
# 2. Tratamento de Dados
# ======================================

# Substituir valores "?" por NaN
df.replace('?', pd.NA, inplace=True)

# Remover linhas com valores ausentes
df.dropna(inplace=True)

# LabelEncoder para transformar atributos categóricos em números
encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Separar features e target
X = df.drop('class', axis=1)
y = df['class']  # 0 = comestível, 1 = venenoso (após encoding)

# ======================================
# 3. Dividir em Treino/Teste
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# 4. Treinar Classificadores
# ======================================

# ---- Classificador 1: SVM ----
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# ---- Classificador 2: Árvore de Decisão ----
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# ======================================
# 5. Avaliar Modelos
# ======================================

def avaliar_modelo(nome, y_test, y_pred):
    print(f"\n=== RESULTADOS: {nome} ===")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


avaliar_modelo("SVM", y_test, svm_pred)
avaliar_modelo("Árvore de Decisão", y_test, tree_pred)
