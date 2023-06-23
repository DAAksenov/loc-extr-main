import numpy as np
from copy import deepcopy
from itertools import product

MAX_MODE = 'MAX'
MIN_MODE = 'MIN'


class SimplexMethod:
    """
    Класс, реализующий симплекс-метод и дальнейшее решение по методу ветвей и границ.
    
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
        self.indent = 0
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
            row = self.get_negative_b_row()  # ищем строку, в которой находятся отрицательное b

            if row == -1:  # если не нашли такую строку
                return True  # то всё хорошо

            column = self.get_negative_b_column(row)  # ищем разрешающий столбец

            if column == -1:
                return False  # не удалось удалить

            self.gauss(row, column)  # выполняем исключение гаусса
            self.calculate_f()
            self.print_table()


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
        Функция, решающая симплекс-таблицу и выводящая последовательность итераций.

        Parameters
        -----------
        iteration: int
        Номер итерации.
        
        Returns
        -----------
        True, False: bool
        
        """
        self.calculate_f()

        if debug:
            print('\nIteration 0')
            self.print_table()

        if not self.remove_negative_b():
            print(self.indent, 'Solve does not exist')
            return False

        iteration = 1

        while True:
            self.calculate_f()

            if debug:
                print('\nIteration', iteration)
                self.print_table()

            if all(fi >= 0 if self.mode == MAX_MODE else fi <= 0 for fi in self.f[:-1]):  # если план оптимален
                break  # то завершаем работу

            column = (np.argmin if self.mode == MAX_MODE else np.argmax)(self.f[:-1])  # получаем разрешающий столбец
            q = self.get_relations(column)  # получаем симплекс-отношения для найденного столбца

            if all(qi == np.inf for qi in q):  # если не удалось найти разрешающую строку
                print('Solve does not exist')  # сообщаем, что решения нет
                return False

            self.gauss(np.argmin(q), column)  # выполняем исключение гаусса
            iteration += 1

        return True  # решение есть

    # получение целочисленных решений
    def solve_integer(self, debug=False, depth=0):
        """
        Функция, получающая целочисленные решения путем разбиения задачи на две ветки.
        
        """

        self.indent = '    ' * depth
        print(self.indent + 'Start solving task:')
        self.print_task()

        # если решение не было найдено, то добавляем пустое решение
        if not self.solve(debug):
            return []

        print(self.indent, "Solve:", self.get_solve()[:self.main_variables_count], ', F:', self.f[-1], '\n')
        x = self.get_solve()
        real_index = self.get_first_real(x)

        # если решение содержало только целые числа, то возвращаем решение
        if real_index == -1:
            return [(x, self.f[-1])]

        b = int(x[real_index])  # получаем новое условие

        # добавляем условия
        task1_a = np.append(self.task['a'], np.array([[1 if i == real_index else 0 for i in range(self.main_variables_count)]]), 0)
        task1_b = np.append(self.task['b'], b)

        task2_a = np.append(self.task['a'], np.array([[-1 if i == real_index else 0 for i in range(self.main_variables_count)]]), 0)
        task2_b = np.append(self.task['b'], -(b + 1))

        # разбиваем задачу на 2 ветки решения
        simplex1 = SimplexMethod(self.task['c'], task1_a, task1_b, self.mode)
        simplex2 = SimplexMethod(self.task['c'], task2_a, task2_b, self.mode)

        print(self.indent, 'Divide to tasks: x%d <= %.3f and x%d >= %.3f' % (real_index + 1, b, real_index + 1, b + 1))

        # запускаем решение для каждой ветки
        solve1 = simplex1.solve_integer(debug, depth + 1)
        solve2 = simplex2.solve_integer(debug, depth + 1)

        return solve1 + solve2

    # вывод лучшего решения
    def print_best_solve(self, solves):
        sign = 1 if self.mode == MAX_MODE else -1
        imax = 0

        for i, solve in enumerate(solves):
            if solve[1] * sign > solves[imax][1] * sign:
                imax = i

        x, f = solves[imax]
        return x[:self.main_variables_count], f

    # проверка выполнения ограничений
    def check_solve(self, solve):
        """
        Функция, выполняющая проверку выполнения ограничений.
        
        Returns
        -----------
        True, False: bool
        
        """
        
        for row, bi in zip(self.task['a'], self.task['b']):
            if sum(ai * xi for ai, xi in zip(row, solve)) > bi:
                return False

        return True

    # перебор
    def solve_integer_bruteforce(self, x_max=20):
        solves = []
        print('Valid solutions (bruteforce):')

        # идём по всем наборам x
        for x in product(range(x_max), repeat=self.main_variables_count):
            if self.check_solve(x):  # если решение подходит
                f = sum(xi * ci for xi, ci in zip(x, self.task['c']))
                print(x[:self.main_variables_count], 'F:', f)
                solves.append((x, f))

        self.print_best_solve(solves)

    # вывод симплекс-таблицы
    def print_table(self):
        """
        Функция, выводящая симплекс-таблицы на каждой итерации.
        
        """
        print('     |' + ''.join(['   x%-3d |' % (i + 1) for i in range(self.variables_count)]) + '    b   |')

        for i in range(self.restrictions_count):
            print('%4s |' % ('x' + str(self.basis[i] + 1)) + ''.join([' %6.2f |' % aij for j, aij in enumerate(self.table[i])]))

        print('   F |' + ''.join([' %6.2f |' % aij for aij in self.f]))
        print('   x |' + ''.join([' %6.2f |' % xi for xi in self.get_solve()]))

    # вывод коэффициента
    def print_coef(self, ai, i):
        if ai == 1:
            return 'x%d' % (i + 1)

        if ai == -1:
            return '-x%d' % (i + 1)

        return '%.2fx%d' % (ai, i + 1)

    # вывод задачи
    def print_task(self, full = False):
        print(self.indent, ' + '.join(['%.2fx%d' % (ci, i + 1) for i, ci in enumerate(self.c[:self.main_variables_count]) if ci != 0]), '-> ', self.mode)

        for row in self.table:
            if full:
                print(self.indent, ' + '.join([self.print_coef(ai, i) for i, ai in enumerate(row[:self.variables_count]) if ai != 0]), '=', row[-1])
            else:
                print(self.indent, ' + '.join([self.print_coef(ai, i) for i, ai in enumerate(row[:self.main_variables_count]) if ai != 0]), '<=', row[-1])

def bb(c,a,b, mode = "MIN"):
    """
    Функции для считывания введенной функции и ограничений и решающая задачу методом ветвей и границ.
    
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

    simplex.solve_integer_bruteforce()

    print("\n\nSolve with branch and bound:")
    solves = simplex.solve_integer(True)
    return simplex.print_best_solve(solves)