import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Función para cargar datos desde un archivo CSV
def load_data(filename):
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}'")
        return None
    return data

# Cargar datos
data = load_data('machine failure.csv')
if data is not None:
   
    Relevant=data[['Machine failure', 'Type','Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','TWF','HDF', 'PWF','OSF', 'RNF']]
    MachinesL = Relevant[Relevant["Type"] == "L"]
    MachinesM = Relevant[Relevant["Type"] == "M"]
    MachinesH = Relevant[Relevant["Type"] == "H"]
       
    
    print(Relevant.head(5))
    print(Relevant.value_counts())
    print(Relevant['Type'].value_counts())
    selected_featuresL =  MachinesL .select_dtypes(include=['int', 'float'])
    correlation_matrix_selected_L = selected_featuresL.corr()
    
    selected_featuresM =  MachinesM.select_dtypes(include=['int', 'float'])
    correlation_matrix_selected_M = selected_featuresM.corr()
    
    selected_featuresH =  MachinesM.select_dtypes(include=['int', 'float'])
    correlation_matrix_selected_H = selected_featuresH.corr()
    
    
    print(MachinesL.head(5))
    print(MachinesM.head(5))
    print(MachinesH.head(5))
    
    #MachinesH
    #MachinesH    

    # Crear mapa de calor
    
    #heatmap 01 

    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(   correlation_matrix_selected_L, annot=True, cmap='cividis', fmt=".2f", linewidths=.5)
    # Rotar los tick labels del eje x en 45 grados
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30,fontname='Arial', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(),fontname='Arial', fontsize=10)
    plt.title('Matriz de Correlación: correlation_matrix_selected_L')
    plt.savefig('correlation_matrix_selected_L.jpg')
    plt.show()


    #heatmap 02 
    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(   correlation_matrix_selected_M, annot=True, cmap='twilight', fmt=".2f", linewidths=.5)
    # Rotar los tick labels del eje x en 45 grados
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30,fontname='Arial', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(),fontname='Arial', fontsize=10)
    plt.title('Matriz de Correlación: correlation_matrix_selected_M')
    plt.savefig('correlation_matrix_selected_M.jpg')
    plt.show()

    
     #heatmap 03 
    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(correlation_matrix_selected_H, annot=True, cmap='gist_heat', fmt=".2f", linewidths=.5)
    # Rotar los tick labels del eje x en 45 grados
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30,fontname='Arial', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(),fontname='Arial', fontsize=10)
    plt.title('Matriz de Correlación: correlation_matrix_selected_H')
    plt.savefig('correlation_matrix_selected_H.jpg')
    plt.show()

    # Definir transformadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    # Definir preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','TWF','HDF', 'PWF','OSF', 'RNF'])])

    # Definir pipelines
    classifiers = {
        'AdaBoost': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, random_state=1)))]),
        'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                              ('classifier', GradientBoostingClassifier(random_state=1))]),
        'SVM': Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(probability=True))])
    }

    # Definir parámetros para la búsqueda en cuadrícula
    param_grids = {
        'AdaBoost': {'classifier__n_estimators': [50, 100, 150]},
        'Gradient Boosting': {'classifier__n_estimators': [50, 100, 150], 'classifier__learning_rate': [0.01, 0.1, 1]},
        'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
    }

    # Entrenamiento y ajuste de modelos
    feature_importances = {}
    for name, pipeline in classifiers.items():
        print(f'Entrenando y ajustando {name}...')
        clf = GridSearchCV(pipeline, param_grid=param_grids[name], cv=5, scoring='roc_auc')
        clf.fit(data.drop(columns=['Machine failure']), data['Machine failure'])
        classifiers[name] = clf
        print(f'Mejor puntuación ROC AUC: {clf.best_score_:.2f}')
        print(f'Mejores parámetros: {clf.best_params_}')
        
        # Obtener importancia de características si es posible
        if name != 'SVM':
            feature_importances[name] = clf.best_estimator_.named_steps['classifier'].feature_importances_

    # Comparación de modelos

    plt.figure(figsize=(10, 8))
    for name, clf in classifiers.items():
        fpr, tpr, _ = roc_curve(data['Machine failure'], clf.predict_proba(data.drop(columns=['Machine failure']))[:, 1])
        roc_auc = roc_auc_score(data['Machine failure'], clf.predict_proba(data.drop(columns=['Machine failure']))[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (ROC AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC')
    plt.legend(loc='lower right')
    plt.savefig('Curvas ROC.jpg')
    plt.show()
   
  
# Visualización de Importancia de Características con estilo Seaborn
plt.figure(figsize=(12, 8))
for name, importance in feature_importances.items():
    if len(importance) > 0:
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = np.array(importance)[sorted_indices]
        sorted_columns = np.array(data.drop(columns=['Machine failure']).columns)[sorted_indices]
        sns.barplot(x=sorted_columns, y=sorted_importance, alpha=0.7, label=name)

plt.xlabel('Características')
plt.ylabel('Importancia')
plt.title('Importancia de Características')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('Importancia de Características.jpg')
plt.show()