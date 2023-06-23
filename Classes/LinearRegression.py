from audioop import bias
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures

class LinearModels:
    """
    Класс, создающий модель линейной регрессии

    Attributes
    ----------
    __model: str
        Тип модели {'classic', 'expo', 'poly'}
    __weights: array
        Полученные веса
    X: array
        Массив признаков
    Y: array
        Массив таргетов

    Methods
    -------
    fit(self)
        Обучает модель
        Returns: self
        
    predict(self, X: list or array)
        Предсказывет значения для введеного набора/наборов признаков X
        Returns: array

    visualize(self)
        Строит график полученной линейной модели и наносит на график данные, на которых модель обучалась
        Returns: plotly.graphic_objects.Figure

    analytuc_func(self)
        Возвращает модель а аналитическом виде
        Returns: str

    """
    def __init__(self, model = 'classic'):
        self.__model = model

    def get_weights(self):
        return self.__weights

    def fit(self, X, Y,reg = None, alpha = 0.1, poly_degree = 2):
        assert len(X) == len(Y)

        model = self.__model
        self.X = X 
        self.Y = np.array(Y)
        X = np.c_[np.ones(len(X)), np.array(X)]
        self.n_features = X.shape[1] - 1
        Y = np.array(Y)

        if model == 'poly':
            self.__degree = poly_degree
            X = PolynomialFeatures(degree=self.__degree).fit_transform(np.c_[np.array(self.X)])
        elif model == 'expo':
            Y = np.log(Y)

        if reg == 'l2':
            loss = lambda weights: np.mean((X.dot(weights) - Y)**2) + alpha*np.sum((weights[1:])**2)
            self.__weights = minimize(loss, x0 = X.shape[1]*[0]).x
        elif reg == 'l1':
            loss = lambda weights: np.mean((X.dot(weights) - Y)**2) + alpha*np.sum(np.abs(weights[1:]))
            self.__weights = minimize(loss, x0 = X.shape[1]*[0]).x
        else:
            loss = lambda weights: np.mean((X.dot(weights) - Y)**2)
            self.__weights = minimize(loss, x0 = X.shape[1]*[0]).x
        if model == 'expo':
            self.__weights = np.exp(self.__weights)
        return self
    def predict(self, X):

        if self.__model == 'poly':
            X = PolynomialFeatures(degree=self.__degree).fit_transform(np.c_[np.array(X)])

        elif self.__model == 'expo':
            X = np.c_[np.array(X)]
            return self.__weights[0]*self.__weights[1:]**X
        
        else:
            X = np.c_[np.ones(len(X)), np.array(X)]
            
        return X.dot(self.__weights)

    def visualize(self):
        assert self.get_weights is not None, 'Следует сначала обучить модель'
        assert self.n_features <=2, 'Виузализация недоступна: кол-во регрессоров > 2'

        fig = go.Figure()
        if self.n_features == 1:
            x = np.linspace(np.min(self.X),np.max(self.X))
            y = self.predict(x)
            fig.add_trace(go.Scatter(x = x.reshape(1,-1)[0], y = y.reshape(1,-1)[0], name = 'y_p'))
            fig.add_trace(go.Scatter(x = self.X, y = self.Y, mode = 'markers', name = 'y'))
            fig.update_layout(
                xaxis_title = 'X',
                yaxis_title = 'y'
            )
        else:
            x1, x2 = np.linspace(np.min(self.X),np.max(self.X)), np.linspace(np.min(self.X),np.max(self.X))
            X1, X2 =np.meshgrid(x1, x2)
            y = self.__weights[0] + X1*self.__weights[1] + self.__weights[2]
            fig = go.Figure(data=[go.Surface(
                x = x1,
                y = x2,
                z = y
            )])
            fig.add_trace(go.Scatter3d(
                x = self.X[:,0],
                y = self.X[:,1],
                z = self.Y
            ))
            fig.update_layout(
                width=1000, height=1000,
                margin=dict(l=65, r=50, b=65, t=90))
        return fig

    def analytical_func(self):
        result = 'f(X) = '
        if self.__model == 'classic':
            result += f'{self.__weights[0]}'
            for i in range(1, len(self.__weights)):
                result += f' + {self.__weights[i]}*x{i}'
        elif self.__model == 'poly':
            result = 'Вывод аналитической функции не реализован для полиномиальной модели'
        
        else:
            result += f'{self.__weights[0]}*{self.__weights[1]}**x{1}'
            for i in range(2, len(self.__weights)):
                result += f' + {self.__weights[0]}*{self.__weights[i]}**x{i}'

        return result

            

if __name__ == '__main__':
    

    poly = LinearModels('poly')
    poly.fit(X = list(range(-10,10)),Y = np.array(list(range(-10,10)))**9, poly_degree=9)
    print(poly.get_weights())
    print(poly.predict([5,6]))
    poly.visualize().show()
