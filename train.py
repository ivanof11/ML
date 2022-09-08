import scipy.stats as ss
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



def modelos(X_train, X_test, y_train, y_test):

    models = [] # append all models or predictive models
    fit_results = []
    predict_results = []
    residuo = []
    residuo_results = []
    mae = []
    rmse = []
    dic_residuos = {
        'Linear': [], 'DecisionTree': [], 'RandomForest': [], 'KNeighbors': [],
                                        'GaussianNB': [], 'SVR': [], 'GradientBoosting': []
    }

    models.append(LinearRegression())
    models.append(DecisionTreeRegressor())
    models.append(RandomForestRegressor())
    models.append(KNeighborsRegressor(n_neighbors=3))
    models.append(GaussianNB())
    models.append(SVR(kernel='linear'))
    models.append(GradientBoostingRegressor(n_estimators=3))

    # Entrenamiento de los modelos
    for model in models :
        fit_results.append(model.fit(X_train,y_train))

    cont = 0
    # Predicción
    for fit_result in fit_results :
        predict_results.append(fit_result.predict(X_test))
        # Obtencion del residuo en valor absoluto
        residuo.append(abs(fit_result.predict(X_test) - y_test))
        dic_residuos[list(dic_residuos.keys())[cont]] = fit_result.predict(X_test)
        cont += 1

    # Metricas MAE y RMSE
    for predict_result in predict_results :
        mae.append(mean_absolute_error(y_test, predict_result))
        rmse.append(mean_squared_error(y_test, predict_result, squared=False))
    
    # Residuo
    for res in residuo:
        residuo_results.append(res.sum())

    # Dataframe residuos
    cv_residuo = pd.DataFrame(
        {
            'Modelos': ['Linear', 'DecisionTree', 'RandomForest', 'KNeighbors',
                                        'GaussianNB', 'SVR', 'GradientBoosting'],
            'RMSE': residuo_results
        })
    # Dataframe Metricas
    cv_frame = pd.DataFrame(
        {
        "MAE":mae,
        "RMSE": rmse,
        'Modelos': ['Linear', 'DecisionTree', 'RandomForest', 'KNeighbors',
                                        'GaussianNB', 'SVR', 'GradientBoosting']
        })

    return cv_frame, cv_residuo, dic_residuos

def tts(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    return X_train, X_test, y_train, y_test

def model_variables(df, valores):
    dic_variables={}
    for i in range(1,len(valores),1):
        
        X = df.drop(valores[0:i], axis=1)
        y = df['prices'].copy()

        X_train, X_test, y_train, y_test = tts(X,y)
        
        cv_metricas = pd.DataFrame
        cv_residuo = pd.DataFrame
        dic_residuos = {}
        cv_metricas, cv_residuo, dic_residuos = modelos(X_train, X_test, y_train, y_test)
        dic_variables[i] = cv_residuo.sort_values(by='RMSE', ignore_index=True).head(1)

    return dic_variables

def comprobar():
    return print('funciona por favooooor')
