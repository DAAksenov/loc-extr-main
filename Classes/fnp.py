from sympy import *
import numpy as np
import time
import plotly.graph_objects as go
import pandas as pd

x,y = symbols('x y')
class Extremum:
    """
    Класс для поиска и визуализации экстремумов

    Attributes
    ----------
    x, y: np.array
        Массив X, Y. Пределы этих массивов устанавливает пользователь (limits), по умолчанию пределы: [-5, 5] 
    X, Y: np.array
        Массивы координатных сеток 2-мерного координатного пространства для массивов X и Y. (Преобразованные массивы для работы
        с 2-мерной функцией)
    z: np.array
        Массив значений целевой функции
    symb_1, symb_2: Symbol
        Введенные пользователем символы для работы с библиотекой sympy.
    analytic_func: sympy.core.function.Function
        Функция, введенная пользователем, в виде sympy.Function.
    
    Methods
    -------
    extremums(self)
        Безусловный экстремум. Находит все критические точки и определяет их типы (g(x,y) - отсутсвует).
    lagr(self)
        Условный экстремум. Находит все критические точки и определяет их типы  
    visualize(self)
        Строит график целевой функции и наносит на него найденные критические точки
    """
    def __init__(self, func: str, g = None, limits=None): #если limits is None -> не строить график
        """
        Parametres
        ----------
        variables: str
            Названия переменных
        func: str
            Целевая функция
        limits: list
            Ограничения для переменных

        Немного пояснений:
        У класса два аттрибута, представляющую целевую функцию (func, analytic_func). 
        analytic_func на основе функций sympy не может работать с многомерными данными, а numeric_func на основе функций numpy
        не может работать с символами sympy. Поэтому нужны две эти функции.
        """
        self.limits = limits
        if limits is not None:
            assert limits[0][0] < limits[0][1]
            self.x = np.arange(limits[0][0], limits[0][1] + 1, 0.1)
                    
            assert limits[1][0] < limits[1][1]
            self.y = np.arange(limits[1][0], limits[1][1] + 1, 0.1)

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.numeric_func = lambda x,y: eval(func.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('exp', 'np.exp').replace('pi','np.pi'))
        self.z = self.numeric_func(self.X, self.Y)
        self.analytic_func = eval(func)
        self.g = eval(g) if g is not None else None
        self.df_points = None #ans

    def extremums(self):
        """g(x) - ограничивающая функция отсутсвует
        Находит все критические точки и опредляет их типы

        Returns:
            pandas.Dataframe
        """
        self.df_points = pd.DataFrame(columns=['x', 'y', 'z', 'type']) 

        diff_x = self.analytic_func.diff(x)
        diff_y = self.analytic_func.diff(y)
        critical_values = solve([diff_x, diff_y], [x, y], dict = True)

        diff_xx = diff_x.diff(x)
        diff_xy = diff_x.diff(y)
        diff_yy = diff_y.diff(y)

        func_silv = diff_xx * diff_yy - diff_xy**2
        
        for point in critical_values:
            A = diff_xx.subs(x, point[x]).subs(y, point[y])
            val = func_silv.subs(x, point[x]).subs(y, point[y])
            row = {'x': float(point[x]), 'y': float(point[y]), 'z': float(self.analytic_func.subs(x, point[x]).subs(y, point[y]))}
            if val == 0:
                row['type'] = 'Требуется доп. исследование'
            elif val < 0:
                row['type'] = 'Седловая точка'
            else:
                if A > 0:
                    row['type'] = 'Min'
                else:
                    row['type'] = 'Max'
            self.df_points = self.df_points.append(row, ignore_index = True)
            if self.limits is not None:
                self.df_points = self.df_points[self.df_points.x >= self.limits[0][0]]
                self.df_points = self.df_points[self.df_points.x <= self.limits[0][1]]
                
                self.df_points = self.df_points[self.df_points.y >= self.limits[1][0]]
                self.df_points = self.df_points[self.df_points.y <= self.limits[1][1]]                
                
        return self.df_points

    def lagr(self):
        """Задача условного экстремума
        Находит все критические точки и опредляет их типы

        Returns:
            pandas.DataFrame
        """
        w = Symbol('w')
        L = self.analytic_func + w*self.g
        L_x, L_y= L.diff(x), L.diff(y)
        g_x, g_y = self.g.diff(x), self.g.diff(y)
        L_xx, L_yy, L_xy = L_x.diff(x), L_y.diff(y), L_y.diff(x)

        critical_values = solve([L_x, L_y, self.g], [x, y, w], dict = True) # [{x: 0, y:5}, {x:3, y:6}]
        self.df_points = pd.DataFrame(columns=['x', 'y', 'z', 'type'])
        f_subs = lambda f: f.subs([(x, point[x]), (y, point[y]), (w, point[w])])
        for point in critical_values:
            row = {'x': float(point[x]), 'y': float(point[y]), 'z': float(f_subs(self.analytic_func))}
            A = np.linalg.det(np.array([
                [0, f_subs(g_x), f_subs(g_y)],
                [f_subs(g_x), f_subs(L_xx), f_subs(L_xy)],
                [f_subs(g_y), f_subs(L_xy), f_subs(L_yy)]
            ], dtype=float))
            if A > 0:
                row['type'] = 'Max'
            elif A < 0:
                row['type'] = 'Min'
            else:
                row['type'] = 'Седловая точка'
            self.df_points = self.df_points.append(row, ignore_index = True)
           
        if self.limits is not None:
            self.df_points = self.df_points[self.df_points.x >= self.limits[0][0]]
            self.df_points = self.df_points[self.df_points.x <= self.limits[0][1]]
                
            self.df_points = self.df_points[self.df_points.y >= self.limits[1][0]]
            self.df_points = self.df_points[self.df_points.y <= self.limits[1][1]]
        return self.df_points
           
    def visualize(self):
        """Строит график целевой функции и наносит на него найденные критические точки
        """
        fig = go.Figure(data=[go.Surface(
        z=self.z, 
        x=self.x, 
        y=self.y, 
        name = str(self.analytic_func), 
        showlegend = True,
        showscale = False)])
        for  point_type in set(self.df_points.type):
            z = self.df_points[self.df_points.type == point_type].z
            x = self.df_points[self.df_points.type == point_type].x
            y = self.df_points[self.df_points.type == point_type].y
            fig.add_trace(go.Scatter3d(
                z = z, 
                x = x, 
                y= y, 
                name = point_type, 
                showlegend = True)) 
        fig.update_layout(title='z(x,y) = ' +str(self.analytic_func),
                  width=1000, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90))
        return fig

    def gradient(self, vector):
        """
        Получение градиета
        """
        diff_x = self.analytic_func.diff(x).subs([(x, vector[0]), (y, vector[1])])
        diff_y = self.analytic_func.diff(y).subs([(x, vector[0]), (y, vector[1])])
        return np.array([diff_x, diff_y], dtype = float)


    def gradient_descent(self, learn_rate = 0.1, start = [1,1], n_iter=100, tolerance=1e-06):
        """
        Реализация градиентного спуска
        """
        vector = np.array(start, dtype=float)
        for _ in range(n_iter):
            diff = -learn_rate*self.gradient(vector)
            if np.all(np.abs(diff) <= tolerance):
                break
            vector += diff
        return vector

    def time_compare(self, learn_rate, n_iter):
        result ={}
        t = time.time()
        extr = self.extremums()
        result['Classic'] = time.time() - t

        t= time.time()
        extr_grad = self.gradient_descent(learn_rate = learn_rate, n_iter = n_iter)
        result['Grad'] = time.time() - t

        return pd.DataFrame(result, index = [1])

def array_to_df(array):
    x, y = array
    return pd.DataFrame({'x': x, 'y': y}, index = [1])

if __name__ == '__main__': 
    Example1 = Extremum('x y', lambda x, y:y*(x**2)+x*(y**3) - x*y,limits=[[-10, 10], [-1, 1]])
    print(Example1.gradient_descent())
