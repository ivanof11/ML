import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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

def pintar(df, x):
    '''
    Devuelve el valor del metodo shapiro

        Parameters:
                x: variable a comparar
                df: DataSet
    '''
    df
    plt.figure(figsize=(22, 5), facecolor='w')
    plt.subplot(131)
    plt.scatter(df.prices, df[x])
    plt.subplot(132)
    plot = df[x].value_counts().plot(kind='bar')
    plt.subplot(133)
    df[x].value_counts().plot(kind='pie', autopct='%.2f')
    plt.suptitle(x)
    plt.show()


def encoder(df):
    encoder = LabelEncoder()
    
    df['district'] = encoder.fit_transform(df['district'])
    df['neighborhood'] = encoder.fit_transform(df['neighborhood'])
    df['condition'] = encoder.fit_transform(df['condition'])
    df['type'] = encoder.fit_transform(df['type'])
    df['lift'] = encoder.fit_transform(df['lift'])
    df['views'] = encoder.fit_transform(df['views'])
    df['floor'] = encoder.fit_transform(df['floor'])

    return df


def pintarResiduos(cv_residuo):
    
    cv_residuo = cv_residuo.sort_values('RMSE', ascending=False)

    plt.figure(figsize=(6, 3.84), facecolor='w')
    plt.hlines(cv_residuo.Modelos, xmin=0, xmax=cv_residuo.RMSE)
    plt.plot(cv_residuo.RMSE, cv_residuo.Modelos, "o", color='black')
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.title('Comparación de error de test modelos'),
    plt.xlabel('Test RMSE')

def pintarPrediccion(dic, y_test, x):
        plt.figure(figsize=(20,5),facecolor='w')
        sns.distplot(y_test, color='r')
        sns.distplot(dic[x], color='g')


def comprobar():
    return print('funciona por favooooor')
