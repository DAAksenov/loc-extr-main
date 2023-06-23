#Градиентные методы
import numpy as np
import scipy
from scipy.optimize import minimize_scalar, minimize, approx_fprime
import sympy
from numpy import *
import time
import pandas as pd
from streamlit import write
from sympy import dict_merge, solve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GradientMethod:
    """
    Класс реализующий градиентные методы

    Заметки: вместо функции нескольких переменных f(x1, x2, x3,..., xn) лучше использовать f(X), где X: array.
    Убрать eval из класса (обрабатывать вводимые пользователем данные отдельной функцией). Эти исправления расширят возможности
    класса. 

    Attributes
    ----------
    func: lambda
        Оптимизируемая функция
    x0: str
        Начальная точка
    n_variables: int
        Кол-во переменных
    max_iterations: int
        Кол-во итераций
    eps: float
        Точность
    SIR: bool
        Сохранить промежуточные результаты
    PIR: bool
        Вывести промежуточные результаты
    lr: float
        Скорость обучения
    
    Methods
    -------
    minimize2(self, method: {
        'Градиентный спуск с постоянным шагом',
        'Градиентный спуск с дроблением шага',
        'Метод наискорейшего спуска',
        'Метод сопряженных градиентов'
    })
        Оптимизирует целевую функцию выбранным методом
        Returns: {'X': array, 'f(X)': float, 'Кол-во итераций': int}
    function(self, x):
        f(x1, x2, x3, x4) -> f(X)
    visualize(self):
        Строит график сходимости
        Returns: plotly.go.Figure
    
    """
    def __init__(self, func, x0, max_iterations = 500, eps = 1e-5, SIR = False, PIR = False, lr = 0.2):
        self.func = func
        self.x0 = x0
        self.n_variables = len(x0)
        self.max_iterations = max_iterations
        self.eps = eps
        self.SIR = SIR
        self.PIR = PIR
        self.lr = lr
        
    def function(self, x):
        return self.func(*x)

    def gs_fixed_rate(self):
        x = self.x0
        f = self.function(x)
        self.dataset = {'iter': [0], 'x': [x], 'f': [f], 'diff': [0]}
        try:
            for i in range(1, self.max_iterations):
                grad = approx_fprime(x,self.function,epsilon = 1e-7)
                diff = -self.lr*grad
                if np.all(np.abs(diff) <= self.eps):
                    break
                x += diff
                f = self.function(x)
                if self.PIR: #Вывод промежуточных результатов, работает долько для сайта
                    row = {'iter': i, 'diff': np.mean(np.abs(grad[:self.n_variables])), 'X': x[:self.n_variables], 'f(X)': round(f, 4)}
                    write(row)

                self.dataset['iter'].append(i)
                self.dataset['x'].append(x)
                self.dataset['f'].append(f)
                self.dataset['diff'].append(diff)

            if i == self.max_iterations:
                return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 1}
            else:
                return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 0}
        except Exception as e:
            return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 2, 'err': e}


    def gs_splitted_rate(self):
        lr = self.lr
        x = self.x0
        f = self.function(x)
        self.dataset = {'iter': [0], 'x': [x], 'f': [f], 'diff': [0]}
        try:
            for i in range(1, self.max_iterations):
                f0 = self.function(x)
                grad = approx_fprime(x,self.function,epsilon = 1e-7)
                diff = -lr*grad

                if np.all(np.abs(diff) <= self.eps):
                    break
                x += diff
                f = self.function(x)
               
                if self.PIR: #Вывод промежуточных результатов, работает долько для сайта
                    row = {'iter': i, 'diff': np.mean(np.abs(grad[:self.n_variables])), 'X': x[:self.n_variables], 'f(X)': round(f, 4)}
                    write(row)

                self.dataset['iter'].append(i)
                self.dataset['x'].append(x)
                self.dataset['f'].append(f)
                self.dataset['diff'].append(diff)

                if f0 > f:
                    lr *= 1.25
                else:
                    lr /= 1.25
            if i == self.max_iterations:
                return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 1}
            else:
                return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 0}
        except Exception as e:
            return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 2, 'err': e}

    def gs_optimal_rate(self):
        x = self.x0
        f = self.function(x)
        self.dataset = {'iter': [0], 'x': [x], 'f': [f], 'diff': [0]}
        try:
            for i in range(1, self.max_iterations):
                f0 = self.function(x)
                grad = approx_fprime(x,self.function,epsilon = 1e-7)
                lr = minimize_scalar(fun = lambda lr: self.function(x - lr*grad), method='brent').x
                diff = -lr*grad

                if np.all(np.abs(diff) <= self.eps):
                    break
                x += diff
                f = self.function(x)
                if self.PIR: #Вывод промежуточных результатов, работает долько для сайта
                    row = {'iter': i, 'diff': np.mean(np.abs(grad[:self.n_variables])), 'X': x[:self.n_variables], 'f(X)': round(f, 4)}
                    write(row)

                self.dataset['iter'].append(i)
                self.dataset['x'].append(x)
                self.dataset['f'].append(f)
                self.dataset['diff'].append(diff)
            if i == self.max_iterations:
                return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 1}
            else:
                return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 0}
        except Exception as e:
            return {'X': x[:self.n_variables], 'f(X)': round(f, 4), 'Кол-во итераций': i, 'Отчет о работе алгоритма': 2, 'err': e}

    def newton_cg(self):
        """
        Препод сказал, больше не нужно реализовывать функции...
        """
        grad = lambda x: approx_fprime(x,self.function,epsilon = 1e-7)
        res = minimize(fun = self.function, x0 = self.x0, method='Newton-CG', jac = grad,options= {'maxiter': self.max_iterations, 'xtol': self.eps})
        return {'X': res.x[:self.n_variables], 'f(X)': res.fun, 'Кол-во итераций': res.nit}

    def minimize2(self, method = 'Метод сопряженных градиентов'):
        func_dict = {
            'Градиентный спуск с постоянным шагом': self.gs_fixed_rate,
            'Градиентный спуск с дроблением шага': self.gs_splitted_rate,
            'Метод наискорейшего спуска': self.gs_optimal_rate,
            'Метод сопряженных градиентов': self.newton_cg
        }
        return func_dict[method]()
    
    def visualize_contour(self):
        assert self.n_variables == 2
        min_x = np.min(self.dataset['x']) - 0.1
        max_x = np.max(self.dataset['x'])+ 0.1

        x_axis = np.linspace(min_x, max_x)
        y_axis = np.linspace(min_x, max_x)

        z_axis = []
        for x in x_axis:
            z_axis_i = []
            for y in y_axis:
                z_axis_i.append(self.function([x, y]))
            z_axis.append(z_axis_i)
        
        contour = go.Contour(x=x_axis, y=y_axis, z=np.transpose(z_axis), name='f(x, y)', colorscale='ice')

        X = np.array(self.dataset['x'])
        descent_way = go.Scatter(x = X[:,0], y = X[:,1], mode = 'lines+markers')
        fig = go.Figure(data = [contour, descent_way])
        return fig

    def visualize_convergence(self):
        assert self.n_variables == 2
        x = self.dataset['iter']
        assert len(x) > 1, 'Кол-во итераций недостаточно для постройки графика сходимости'
        y_grad = np.sum((np.array(self.dataset['diff']))**2)**0.5
        y_func = self.dataset['f']
        fig = make_subplots(cols = 2)

        fig.add_trace(go.Scatter(x = x, y = y_func, name = 'f(X)'), row = 1, col = 1)

        fig.add_trace(go.Scatter(x = x, y = y_grad, name = '|| g(X) ||'), row = 1, col = 2)
        
        fig.update_layout(
            title = 'Значение функции и l2-норма градиента каждую итерацию',
            xaxis_title = 'Итерации',
            yaxis_title = 'f(X) or || g(X) ||')        
        return fig

def test(func: str, x0: str, max_iterations: int, lr: float):
    obj = GradientMethod(func=func, x0=x0, max_iterations=max_iterations, lr = lr)
    func_dict = {
            'Градиентный спуск с постоянным шагом': obj.gs_fixed_rate,
            'Градиентный спуск с дроблением шага': obj.gs_splitted_rate,
            'Метод наискорейшего спуска': obj.gs_optimal_rate,
            'Метод сопряженных градиентов': obj.newton_cg
        }

    result = pd.DataFrame(
        columns = func_dict.keys(), index = ['Полученное решение', 'Время выполнения', 'Кол-во итераций'])

    for method in func_dict:
        start_time = time.time()
        method_result = func_dict[method]()
        exec_time = time.time() - start_time
        result[method]['Полученное решение'] = method_result['f(X)']
        result[method]['Время выполнения'] = exec_time
        result[method]['Кол-во итераций'] = method_result['Кол-во итераций'] 
    return result

    

if __name__ == '__main__':
    obj = GradientMethod(func = lambda x1, x2:x1**2 + (x2+5)**2, x0 = '[10,10]', max_iterations = 100, eps = 1e-5, SIR = True, PIR = True, lr = 0.1)
    print(obj.minimize2('Градиентный спуск с постоянным шагом'))