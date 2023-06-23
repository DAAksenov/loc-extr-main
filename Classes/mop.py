# Методы одномерной оптимизации

import numpy as np
import sympy as sp
from numpy import *
import time
import pandas as pd
from streamlit import write
from sympy import dict_merge, solve
import plotly.graph_objects as go

def grad(f,x): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    h = np.cbrt(np.finfo(float).eps)
    d = 0
    nabla = np.zeros(d)
    for i in range(d): 
        x_for = np.copy(x) 
        x_back = np.copy(x)
        x_for[i] += h 
        x_back[i] -= h 
        nabla[i] = (f(x_for) - f(x_back))/(2*h) 
    return nabla 

def line_search(f,x,p,nabla, c1, c2):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = c1 
    c2 = c2 
    fx = f(x)
    x_new = x + a * p 
    nabla_new = grad(f,x_new)
    while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
        a *= 0.5
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
    return a

class Extremum_1d:
    """
    Класс, включающий в себя методы одномерной оптимизации

    Methods
    -------
    gss(self)
        Нахождение минимума с помощью метода золотого сечений
        Returns: {
            'x': найденный минимум, 
            'f(x)': значение функции в точке минимума, 
            'Величина исследуемого интервала': ...,
            'Отчет о работе алгоритма': ...}
        
    quadratic_approximation(self)
        Нахождение минимума с помощью метода парабол
        Returns: {
            'x': найденный минимум, 
            'f(x)': значение функции в точке минимума, 
            'Величина исследуемого интервала': ...,
            'Отчет о работе алгоритма': ...}
        
    brent(self)
        Нахождение минимума с помощью метода Брента
        Returns: {
            'x': найденный минимум, 
            'f(x)': значение функции в точке минимума, 
            'Величина исследуемого интервала': ...,
            'Отчет о работе алгоритма': ...}

    BFGS(self)
        Нахождение минимума с помощью метода BFGS
        Returns: {
            'x': найденный минимум, 
            'f(x)': значение функции в точке минимума, 
            'Величина исследуемого интервала': ...,
            'Отчет о работе алгоритма': ...}
    
    plot(self)
        Построение графика рассеивания (только для 2-х признаков)
        Returns: plotly.graphic_objects.Figure


    """
    def __init__(self, 
                func: str,
                a,
                b,
                eps = 10**(-5),
                max_iter = 500,
                print_intermediate_results = False,
                save_intermediate_results = False,
                x0 = 1,
                max_x = 100,
                c1 = 1e-4,
                c2 = 0.1):
        self.func = lambda x: eval(func)
        self.a = a
        self.b = b
        self.eps = eps
        self.max_iter = max_iter
        self.PIR = print_intermediate_results
        self.SIR = save_intermediate_results
        self.x0 = x0
        self.max_x = max_x
        self.c1 = c1,
        self.c2 = c2
        self.results = None


    def gss(self):
        a,b = self.a, self.b
        gr = (np.sqrt(5) + 1) / 2
        f = self.func
        self.results = pd.DataFrame(columns=['x', 'f(x)', 'Величина исследуемого интервала', 'Отчет о работе алгоритма'])

        c = b - (b - a) / gr
        d = a + (b - a) / gr
        n_iter = 0
        while abs(b - a)/2 > self.eps and n_iter < self.max_iter:
            if f(c) < f(d):
                b = d
            else:
                a = c

            # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            n_iter += 1
            if self.PIR:
                write(f'x = {(a + b) / 2}, f(x) = {f((a + b) / 2)}, iter = {n_iter}')
            if self.SIR and n_iter < self.max_iter and abs(a - b)/2 > self.eps:
               self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 3}, ignore_index=True)

        if n_iter == self.max_iter:
            self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 1}, ignore_index=True)
        else:
            self.results = self.results.append({'x': (a+b)/2, 'f(x)': f((a+b)/2), 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 0}, ignore_index=True)
        return self.results

    def quadratic_approximation(self):
        try:
            func = self.func
            a, b, c = self.a, self.b, (self.a + self.b) / 2
            f0, f1, f2 = func(a), func(b), func(c)
            f_x = {a: f0, b: f1, c: f2}
            self.results = pd.DataFrame(columns=['x', 'f(x)', 'Величина исследуемого интервала', 'Отчет о работе алгоритма'])
            c, b, a = sorted([a, b, c], key=lambda x: f_x[x])
            n_iter = 0
            while n_iter < self.max_iter and abs(b - c)/2 > self.eps:
                f0, f1, f2 = f_x[a], f_x[b], f_x[c]
                p = (b - c) ** 2 * (f2 - f0) + (a - c) ** 2 * (f1 - f2)
                q = 2 * ((b - c) * (f2 - f0) + (a - c) * (f1 - f2))
                assert p != 0
                assert q != 0

                x_new = c + p / q
                assert self.a <= x_new <= self.b

                f_new = func(x_new)
                f_x[x_new] = f_new
                previous_xs = [a, b, c]

                if f_new < f2:
                    a, f0 = b, f1
                    b, f1 = x_new, f_new
                    c, f2 = x_new, f_new

                elif f_new < f1:
                    a, f0 = b, f1
                    b, f1 = x_new, f_new

                elif f_new < f0:
                    a, f0 = x_new, f_new
                
                n_iter += 1

                if self.PIR:
                    write(f'x = {c}, f(x) = {f2}, iter = {n_iter}')

                if self.SIR and n_iter < self.max_iter and abs(b - c)/2 > self.eps:
                        self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 3}, ignore_index=True)        
            
            if n_iter == self.max_iter:
                self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 1}, ignore_index=True)
            else:
                self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 0}, ignore_index=True)
            return self.results
        except Exception as e:
            self.results = self.results.append({'x': c, 'f(x)': f2, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 2}, ignore_index=True)
            return self.results
    
    def brent(self):

        a,b = self.a, self.b
        gr = (3 - 5 ** 0.5) / 2
        f = self.func
        n_iter = 0
        self.results = pd.DataFrame(columns=['x', 'f(x)', 'Величина исследуемого интервала', 'Отчет о работе алгоритма'])

        x_largest = x_middle = x_least = a + gr * (b - a)
        f_largest = f_middle = f_least = f(x_least)
        remainder = 0.0
        middle_point = (a + b) / 2
        tolerance = self.eps * abs(x_least) + 1e-9
        while n_iter < self.max_iter and abs(x_least - middle_point) > 2 * tolerance - (b - a) / 2:
            middle_point = (a + b) / 2
            tolerance = self.eps * abs(x_least) + 1e-9

            p = q = previous_remainder = 0
            if abs(remainder) > tolerance:
                p = ((x_least - x_largest) ** 2 * (f_least - f_middle) -(x_least - x_middle) ** 2 * (f_least - f_largest))
                q = 2 * ((x_least - x_largest) * (f_least - f_middle) -(x_least - x_middle) * (f_least - f_largest))
            if q > 0:
                p = -p
            else:
                q = -q
            previous_remainder = remainder

            if abs(p) < 0.5 * abs(q * previous_remainder) and a * q < x_least * q + p < b * q:
                remainder = p / q
                x_new = x_least + remainder

                if x_new - a < 2 * tolerance or b - x_new < 2 * tolerance:
                    if x_least < middle_point:
                        remainder = tolerance
                    else:
                        remainder = -tolerance
            
            else:
                if x_least < middle_point:
                    remainder = (b - x_least) * gr
                else:
                    remainder = (a - x_least) * gr

            if abs(remainder) > tolerance:
                x_new = x_least + remainder
            elif remainder > 0:
                x_new = x_least + tolerance
            else:
                x_new = x_least - tolerance

            f_new = f(x_new)

            if f_new <= f_least:
                if x_new < x_least:
                    b = x_least
                else:
                    a = x_least
                
                x_largest = x_middle
                f_largest = f_middle

                x_middle = x_least
                f_middle = f_least

                x_least = x_new
                f_least = f_new

            else:
                if x_new < x_least:
                    a = x_new
                else:
                    b = x_new
                if f_new <= f_middle:
                    x_largest = x_middle
                    f_largest = f_middle

                    x_middle = x_new
                    f_middle = f_new
                elif f_new <= f_largest:
                    x_largest = x_new
                    f_largest = f_new
            n_iter += 1

            if self.PIR:
                write(f'x = {x_least}, f(x) = {f_least}, iter = {n_iter}')
            
            if self.SIR and n_iter < self.max_iter and abs(x_least - middle_point) > 2 * tolerance - (b - a) / 2:
                self.results = self.results.append({'x': x_least, 'f(x)': f_least, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 3}, ignore_index=True)
        
        if n_iter == self.max_iter:
            self.results = self.results.append({'x': x_least, 'f(x)': f_least, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 1}, ignore_index=True)
        else:
            self.results = self.results.append({'x': x_least, 'f(x)': f_least, 'Величина исследуемого интервала':abs(a-b), 'Отчет о работе алгоритма': 0}, ignore_index=True)       

        return self.results
    
    def BFGS(self):
        f = self.func
        x0 = self.x0
        d = 0 # dimension of problem 
        nabla = grad(f,x0) # initial gradient 
        H = np.eye(d) # initial hessian
        x = x0
        n_iter = 0 
        self.results = pd.DataFrame(columns=['x', 'f(x)', 'Величина исследуемого интервала', 'Отчет о работе алгоритма'])

        while np.linalg.norm(nabla) > self.eps and self.max_iter > n_iter: # while gradient is positive
            if n_iter > self.max_iter: 
                print('Maximum iterations reached!')
                break
            n_iter += 1
            p = -H@nabla # search direction (Newton Method)
            a = line_search(f,x,p,nabla, self.c1, self.c2) # line search 
            s = a * p 
            x_new = x + a * p 
            nabla_new = grad(f,x_new)
            y = nabla_new - nabla 
            y = np.array([y])
            s = np.array([s])
            y = np.reshape(y,(d,1))
            s = np.reshape(s,(d,1))
            r = 1/(y.T@s)
            li = (np.eye(d)-(r*((s@(y.T)))))
            ri = (np.eye(d)-(r*((y@(s.T)))))
            hess_inter = li@H@ri
            H = hess_inter + (r*((s@(s.T)))) # BFGS Update
            nabla = nabla_new[:] 
            x = x_new[:]
            if self.PIR:
                write(f'x = {x}, f(x) = {f(x)}, iter = {n_iter}')
            if self.SIR and n_iter < self.max_iter and np.linalg.norm(nabla) > self.eps:
                self.results = self.results.append({'x': x, 'f(x)': f(x), 'Величина исследуемого интервала': 'Нет исс, интервала', 'Отчет о работе алгоритма': 3}, ignore_index=True)
        if n_iter == self.max_iter:
            self.results = self.results.append({'x': x, 'f(x)': f(x), 'Величина исследуемого интервала': 'Нет исс, интервала', 'Отчет о работе алгоритма': 1}, ignore_index=True)
        else:
            self.results = self.results.append({'x': x, 'f(x)': f(x), 'Величина исследуемого интервала': 'Нет исс, интервала', 'Отчет о работе алгоритма': 0}, ignore_index=True)
        return self.results
        
class ExtraTasks(Extremum_1d):
    """
    доп задания на сравнение  и визуализацию алгоритмов
    """
    def __init__(self, 
                func: str,
                a,
                b,
                eps = 10**(-5),
                max_iter = 500,
                print_intermediate_results = False,
                save_intermediate_results = True,
                method = 'Метод золотого сечения',
                x0 = 1,
                max_x = 500,
                c1 = 1e-4,
                c2 = 0.1):
        Extremum_1d.__init__(self, 
                func,
                a,
                b,
                eps,
                max_iter,
                print_intermediate_results,
                save_intermediate_results,
                x0,
                c1,
                c2)
        self.dict_method = {
        'Метод золотого сечения': self.gss,
        'Метод парабол': self.quadratic_approximation,
        'Метод Брента': self.brent,
        'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно': self.BFGS}
        self.results = self.dict_method[method]()
        self.method = method
        self.sp_func = lambda x: eval(func.replace('sin', 'sp.sin').replace('cos', 'sp.cos').replace('exp', 'sp.exp').replace('pi','sp.pi').replace('arccos', 'sp.arccos'))

    def q3(self):
        x = np.linspace(self.a, self.b)
        y = self.func(x)
        c = self.results['f(x)'].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='f(x)'))

        for i in range(len(self.results)):
            fig.add_trace(go.Scatter(x=[self.results['x'][i]], y=[self.results['f(x)'][i]],
                    mode='markers',
                    name=f'iter {i}'))
        return fig

    def q4(self):
        if self.method == 'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно':
            write('ВЫберите любой другой метод')
        x = self.results.index + 1
        y = self.results['Величина исследуемого интервала'].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines'))
        fig.update_layout(title='Сходимость алгоритма',
                   xaxis_title='Итерации',
                   yaxis_title='Величина исследуемого интервала')
        return fig

    def q5(self):
        df = pd.DataFrame(columns= ['Получено решение', 'Время выполнения', 'Кол-во итераций'], index = self.dict_method.keys())
        for method in df.index:
            start_time = time.time()
            result = self.dict_method[method]()
            exec_time = time.time() - start_time

            df.loc[method, 'Получено решение'] = f'x = {result.iloc[[-1], [0]].values[0][0]}, f(x) = {result.iloc[[-1], [1]].values[0][0]}'
            df.loc[method, 'Время выполнения'] = exec_time
            df.loc[method, 'Кол-во итераций'] =  len(result.index) - 1
        return df



            

if __name__ == '__main__':
    print(ExtraTasks('-5*x**5 + 4*x**4 - 12*x**3  + 11*x**2 - 2*x + 1',-0.5,0.5,print_intermediate_results=True, save_intermediate_results=True,method = 'Метод золотого сечения').minimize())