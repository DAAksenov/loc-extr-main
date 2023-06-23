import numpy as np
from copy import deepcopy
from itertools import product
import time
import plotly.graph_objects as go

MAX_MODE = 'MAX'  # режим максимизации
MIN_MODE = 'MIN'  # режим минимизации


class SimplexMethod:
    """
    Класс, реализующий симплекс-метод и дальнейшее решение по методу Гомори.
    
    """

    def __init__(self, c, a, b, mode):
        """
        Функция для инициализации всех переменных.
        
        """

        self.main_variables_count = a.shape[1]  # количество переменных
        self.restrictions_count = a.shape[0]  # количество ограничений
        self.variables_count = self.main_variables_count + self.restrictions_count  # количество переменных
        self.mode = mode  # запоминаем режим работы

        self.c = np.concatenate([c, np.zeros((self.restrictions_count + 1))])  # коэффициенты функции
        self.f = np.zeros((self.variables_count + 1))  # значения функции F
        self.basis = [i + self.main_variables_count for i in range(self.restrictions_count)]  # индексы базисных переменных
        self.task = {'a': deepcopy(a), 'b': deepcopy(b), 'c': deepcopy(c)}  # сохраняем начальную задачу

        self.init_table(a, b)

    # инициализация таблицы
    def init_table(self, a, b):
        """
        Функция для инициализации таблицы.

        Parameters
        -----------
        table: numpy.array
        Таблица с коэффициентами.
        
        """
        self.table = np.zeros((self.restrictions_count, self.variables_count + 1))  # коэффициенты таблицы

        for i in range(self.restrictions_count):
            for j in range(self.main_variables_count):
                self.table[i][j] = a[i][j]

            for j in range(self.restrictions_count):
                self.table[i][j + self.main_variables_count] = int(i == j)

            self.table[i][-1] = b[i]

    # получение строки с максимальным по модулю отрицательным значением b
    def get_negative_b_row(self):
        """
        Функция получения строки с максимальным по модулю отрица-тельным значением b.
        
        Parameters
        -----------
        row: numpy.array
        Строка с коэффициентами.
        
        Returns
        -----------
        row: numpy.array
        Строка с максимальным по модулю отрицательным значением b.

        """
        row = -1

        for i, a_row in enumerate(self.table):
            if a_row[-1] < 0 and (row == -1 or abs(a_row[-1]) > abs(self.table[row][-1])):
                row = i

        return row

    # получение столбца с максимальным по модулю элементом в строке
    def get_negative_b_column(self, row):
        
        """
        Функция получения столбца с максимальным по модулю эле-ментом в строке.

        Parameters
        -----------
        column: numpy.array
        Столбец с коэффициентами.
        
        Returns
        -----------
        column: numpy.array
        Столбец с максимальным по модулю элементом в столбце.
        
        """
        column = -1

        for i, aij in enumerate(self.table[row][:-1]):
            if aij < 0 and (column == -1 or abs(aij) > abs(self.table[row][column])):
                column = i

        return column

    # удаление отрицательных свободных коэффициентов
    def remove_negative_b(self):
        """
        Функция удаления отрицательных свободных коэффициентов.

        Parameters
        -----------
        column: numpy.array
        Столбец с коэффициентами.
        
        row: numpy.array
        Строка с коэффициентами.
        
        """
        while True:
            row = self.get_negative_b_row()  # ищем строку, в которой находится отрицательное b

            if row == -1:  # если не нашли такую строку
                return True  # то всё хорошо

            column = self.get_negative_b_column(row)  # ищем разрешающий столбец

            if column == -1:
                return False  # не удалось удалить

            self.gauss(row, column)  # выполняем исключение гаусса
            self.calculate_f()

    # выполнение шага метода гаусса
    def gauss(self, row, column):
        """
        Функция выполняющая шаг метода Гаусса.

        Parameters
        -----------
        table: numpy.array
        Таблица с коэффициентами.
        
        """
        self.table[row] /= self.table[row][column]

        for i in range(self.restrictions_count):
            if i != row:
                self.table[i] -= self.table[row] * self.table[i][column]

                for j in range(self.variables_count):
                    if abs(self.table[i][j]) < 1e-10:
                        self.table[i][j] = 0

        self.basis[row] = column  # делаем переменную базисной

    # расчёт значений F
    def calculate_f(self):
        """
        Функция, считающая значения заданной функции.

        Parameters
        -----------
        table: numpy.array
        Таблица с коэффициентами.
        
        """
        for i in range(self.variables_count + 1):
            self.f[i] = -self.c[i]

            for j in range(self.restrictions_count):
                self.f[i] += self.c[self.basis[j]] * self.table[j][i]

    # расчёт симплекс-отношений для столбца column
    def get_relations(self, column):
        """
        Функция, делающая расчет симплекс отношений для столбца column.

        Parameters
        -----------
        q: list
        Список симплекс-отношений.
        
        Returns
        -----------
        q: list
        Список со значениями симплекс-отношений.
        
        """

        q = []

        for i in range(self.restrictions_count):
            if self.table[i][column] == 0:
                q.append(np.inf)
            else:
                q_i = self.table[i][-1] / self.table[i][column]
                q.append(q_i if q_i >= 0 else np.inf)

        return q

    # получение решения
    def get_solve(self):
        x = np.zeros((self.variables_count))

        # заполняем решение
        for i in range(self.restrictions_count):
            x[self.basis[i]] = self.table[i][-1]

        return x  # возвращаем полученное решение

    # получение первого вещественного решения
    def get_first_real(self, x):
        return next((i for i, xi in enumerate(x[:self.main_variables_count]) if xi != int(xi)), -1)

    # решение
    def solve(self, debug):
        """
        Функция, решающая симплекс-таблицу и выводящая последова-тельность итераций.

        Parameters
        -----------
        iteration: int
        Номер итерации.
        
        Returns
        -----------
        True, False: bool
        
        """
        self.calculate_f()
        s = ''
        if debug:
            s+='\nIteration 0'
            s += self.print_table()

        if not self.remove_negative_b():
            s+='Solve does not exist'
            return False

        iteration = 1

        while True:
            self.calculate_f()

            if debug:
                s+=f'\nIteration,{iteration}'
                s +=self.print_table()

            if all(fi >= 0 if self.mode == MAX_MODE else fi <= 0 for fi in self.f[:-1]):  # если план оптимален
                break  # то завершаем работу

            column = (np.argmin if self.mode == MAX_MODE else np.argmax)(self.f[:-1])  # получаем разрешающий столбец
            q = self.get_relations(column)  # получаем симплекс-отношения для найденного столбца

            if all(qi == np.inf for qi in q):  # если не удалось найти разрешающую строку
                s+='Solve does not exist'  # сообщаем, что решения нет
                return False

            self.gauss(np.argmin(q), column)  # выполняем исключение гаусса
            iteration += 1

        return True , s # решение есть

    # получение дробной части
    def part(self, a):
        """
        Функция, возвращающая дробную часть числа.

        Returns
        -----------
        a: int
        Дробная часть числа.
        
        """
        if a > 0:
            return a - int(a)

        return int(abs(a)) + 1 + a

    def make_gamori(self, x, real_index, debug):
        """
        Функция, реализующая метод Гомори.

        Parameters
        -----------
        r_main: list
        r_basis: list
        task1_a: np.array()
        task1_b: np.array
        
        Returns
        -----------
        Целочисленные решения.
        
        """
        b = self.part(x[real_index])
        r_main = [0 if i in self.basis else self.part(self.table[real_index][i]) for i in range(self.main_variables_count)]
        r_basis = [0 if i in self.basis else self.part(self.table[real_index][i]) for i in range(self.main_variables_count, self.variables_count)]

        # добавляем условия
        task1_a = np.append(self.task['a'], np.array([r_main]), 0)
        task1_b = np.append(self.task['b'], -b)

        simplex1 = SimplexMethod(self.task['c'], task1_a, task1_b, self.mode)

        for i in range(self.restrictions_count):
            for j in range(self.variables_count):
                simplex1.table[i][j] = self.table[i][j]

            simplex1.table[i][-1] = self.table[i][-1]

        for i in range(self.main_variables_count):
            simplex1.table[-1][i] = -r_main[i]

        for i in range(self.main_variables_count, self.variables_count):
            simplex1.table[-1][i] = -r_basis[i - self.main_variables_count]

        return simplex1.solve_integer(debug)

    # получение целочисленных решений
    def solve_integer(self, debug=False):
        """
        Функция, получающая целочисленные решения.
        
        """
        s = ''
        s+='Start solving task:'
        print(self.print_task())

        # если решение не было найдено, то добавляем пустое решение
        if not self.solve(debug)[0]:
            return []

        s+=f'Solve:, {self.get_solve()[:self.main_variables_count]}, , F:, {self.f[-1]}, \n'
        x = self.get_solve()
        real_index = self.get_first_real(x)

        # если решение содержало только целые числа, то возвращаем решение
        if real_index == -1:
            return [(x, self.f[-1])]
        print(s)
        return self.make_gamori(x, real_index, debug)

    # вывод лучшего решения
    def print_best_solve(self, solves):
        sign = 1 if self.mode == MAX_MODE else -1
        imax = 0

        for i, solve in enumerate(solves):
            if solve[1] * sign > solves[imax][1] * sign:
                imax = i

        x, f = solves[imax]
        print('\nBest solve: ', x[:self.main_variables_count], 'F:', f)
        return x[:self.main_variables_count], f

    # вывод симплекс-таблицы
    def print_table(self):
        """
        Функция, выводящая симплекс-таблицы на каждой итерации.
        
        """
        s = ''
        s+='     |' + ''.join(['   x%-3d |' % (i + 1) for i in range(self.variables_count)]) + '    b   |'

        for i in range(self.restrictions_count):
            s+='%4s |' % ('x' + str(self.basis[i] + 1)) + ''.join([' %6.2f |' % aij for j, aij in enumerate(self.table[i])])

        s+='   F |' + ''.join([' %6.2f |' % aij for aij in self.f])
        #print('   x |' + ''.join([' %6.2f |' % xi for xi in self.get_solve()]))
        return s

    # вывод коэффициента
    def print_coef(self, ai, i):
        if ai == 1:
            return 'x%d' % (i + 1)

        if ai == -1:
            return '-x%d' % (i + 1)

        return '%.2fx%d' % (ai, i + 1)

    # вывод задачи
    def print_task(self, full = True):
        s = ' + '.join(['%.2fx%d' % (ci, i + 1) for i, ci in enumerate(self.c[:self.main_variables_count]) if ci != 0]), '-> ', self.mode

        for row in self.table:
            if full:
                s+=' + '.join([self.print_coef(ai, i) for i, ai in enumerate(row[:self.variables_count]) if ai != 0]), '=', row[-1]
            else:
                s+=' + '.join([self.print_coef(ai, i) for i, ai in enumerate(row[:self.main_variables_count]) if ai != 0]), '<=', row[-1]
        return s


def main(c,a,b, mode = "MIN"):
    """
    Функции для считывания введенной функции и ограничений и решающая задачу методом Гомори.
    Parameters
    -----------
    с: numpy.array
    Коэффициенты заданной функции.
    a: numpy.array
    Коэффициенты заданных ограничений
    b: numpy.array
    Коэффициенты свободных членов ограничений
    mode: str
    {"MIN", "MAX"}
    Returns
    -----------
    Найденную координату и значение функции в этой координате.
    
    """


    simplex = SimplexMethod(c, a, b, mode)

    print("Solve with Gomory method:")
    solves = simplex.solve_integer(True)
    return simplex.print_best_solve(solves)


def visualize(c, a, b, result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[result[0][0]],y=[result[0][1]], showlegend=False, marker = dict(size = 10, color = [0])))
    x2min = 0
    x2max = -np.inf
    for i in range(len(a)):
        g = lambda x1: (b[i] - a[i][0]*x1)/a[i][1]
        x = np.linspace(0, result[0][0] + 2)
        x2 = g(x)
        if np.min(x2) < x2min:
            x2min = np.min(x2)
        if np.max(x2) > x2max:
            x2max = np.max(x2)
        fig.add_trace(go.Scatter(x = x , y = x2, mode = 'lines', showlegend=False))
    x2 = np.linspace(x2min, x2max)
    X, Y = np.meshgrid(x, x2)
    Z = c[0]*X + c[1]*Y
    fig.add_trace(go.Contour(x = x, y = x2, z = Z,  colorscale = 'ice',name = 'Probability', 
 showscale = True
 ))
    return fig