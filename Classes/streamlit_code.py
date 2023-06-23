import streamlit as st
from Classes import fnp, mop, gradient_methods, LinearRegression, Classifier, b_and_b, gamori, tz5, tz8
from numpy import *
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, make_gaussian_quantiles
import torch


def page1():
    st.title('Поиск экстремумов ФНП')
    st.write('Приложение ищет экстремумы функции двух переменных в действительной плоскости')
    st.write("***")

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Ввести нужную функцию в стиле numpy в поле “Введите целевую функцию”.
        Можно использовать функции, реализованные в numpy: np.sin(x) + np.exp(5)""")
        st.write("""2. Выберите тип задачи""")
        st.write("""3. Введите нижнюю границу для x""")
        st.write("""4. Введите верхнюю границу для x""")
        st.write("""5. Введите нижнюю границу для y""")
        st.write("""6. Введите верхнюю границу для y""")
        st.write("""7. Ввести дополнительные параметры""")
        st.write("""8. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация графиков')
        st.write("""1. На экране должна отобразиться таблица критических точек, найденных алгоритмом""")
        st.write(
            """2. Так же должен построиться график целевой функции с критическими точками, если кол-во переменных <= 2, """)

    # Настройка условий
    func = st.sidebar.text_input('Введите целевую функцию', value='y*(x**2)+x*(y**3) - x*y')
    task = st.sidebar.selectbox(label='Выберите тип задачи', options=['Безусловный экстремум', 'Условный экстремум'])
    if task == 'Условный экстремум':
        g = st.sidebar.text_input('Введите ограничивающую функцию', value='x**2+4*y**2 - 1')
    x_d = st.sidebar.number_input('Введите нижнюю границу для x', value=-10)
    x_up = st.sidebar.number_input('Введите верхнюю границу для x', value=10)
    y_d = st.sidebar.number_input('Введите нижнюю границу для y ', value=-1)
    y_up = st.sidebar.number_input('Введите верхнюю границу для y ', value=1)
    limits = [[x_d, x_up], [y_d, y_up]]

    if task == 'Условный экстремум':
        """ После каждого нажатия следующей кнопки (дальше будут кнопки) переменные под блоком кнопки удалятся из памяти,
        поэтому мы определяем объект класса здесь, в дальнейшем он нам пригодится для третьего задания
        """
        Example = fnp.Extremum(func=func, g=g, limits=limits)
    else:
        Example = fnp.Extremum(func=func, limits=limits)

    if st.sidebar.button('Найти экстремумы и построить график'):
        st.header('Задания 1, 2')
        if task == 'Условный экстремум':
            st.dataframe(Example.lagr())
            st.plotly_chart(Example.visualize())
        else:
            st.dataframe(Example.extremums())
            st.plotly_chart(Example.visualize())

    learn_rate = st.sidebar.slider('Выберите learn_rate', min_value=0.00, max_value=1.00, step=0.05)
    n_iter = st.sidebar.slider('Выберите макс кол-во итераций', min_value=0, max_value=200)
    if st.sidebar.button('Сравнить производительность разных алгоритмов'):
        st.header('Задание 3')
        st.dataframe(Example.time_compare(learn_rate, n_iter))
        st.write('Точка минимума/седловая точка, полученная градиентным спуском')
        st.dataframe(fnp.array_to_df(Example.gradient_descent(learn_rate=learn_rate, n_iter=n_iter)))
        st.write('Стац. точки, полученные аналитическим способом')
        st.dataframe(Example.extremums())


def page2():
    st.title('Поиск экстремумов функции одной переменной')
    st.write('Приложение ищет экстремумы функции одной переменной в действительной плоскости')
    st.write("***")

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Ввести нужную функцию в стиле numpy в поле “Введите целевую функцию”.
        Можно использовать функции, реализованные в numpy: np.sin(x) + np.exp(5)""")
        st.write("""2. Выбрать нужный метод""")
        st.write("""3. Ввести нужные границы интервала (a и b)""")
        st.write("""4. Ввести нужную точность """)
        st.write("""5. Ввести максимально возможное количество итераций для цикла""")
        st.write("""6. Выбрать нужно ли выводить промежуточные результаты: x и f(x) в каждой итерации""")
        st.write("""7. Ввести дополнительные параметры""")
        st.write("""8. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация графиков')
        st.write(
            """1. На первом графике (Задание 3) изображена целевая функция, а так же точки(значение функции) на каждой итерации""")
        st.write("""2. На втором графике (Задание 4) построен график сходимости. В данном тз методы основаны на интервалах
        (оптимальная точка находится в этом интервале), с каждой итерацией интервал уменьшается, это изображено на графике""")
    # Настройка условий
    func = st.sidebar.text_input('Введите целевую функцию', value='x**2 - 2*x + 10')
    task = st.sidebar.selectbox(label='Выберите метод',
                                options=['Метод золотого сечения', 'Метод парабол', 'Метод Брента',
                                         'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно'])
    a = st.sidebar.number_input('Введите нижнюю границу интервала', value=-0.5)
    b = st.sidebar.number_input('Введите верхнюю границу интервала', value=0.5)
    eps = st.sidebar.number_input('Введите точность', value=1e-3, min_value=1e-5, max_value=1., step=1e-6,
                                  format='%.6f')
    max_iter = st.sidebar.number_input('Введите макс. кол-во итераций', value=500)
    PIR = st.sidebar.selectbox(label='Вывод промежуточных результатов', options=[False, True])
    SIR = st.sidebar.selectbox(label='Сохранение промежуточных результатов', options=[False, True])
    st.sidebar.write('Параметры для BFGS')
    x0 = st.sidebar.number_input('Введите начальную точку', value=1)
    max_x = st.sidebar.number_input('Введите максимальное значение аргумента функции', value=100)
    c1 = st.sidebar.number_input('Введите параметр для первого условия Вольфе', value=10 ** (-4))
    c2 = st.sidebar.number_input('Введите параметр для второго условия Вольфе', value=0.1)
    Example = mop.Extremum_1d(func=func, a=a, b=b, eps=eps, max_iter=max_iter, print_intermediate_results=PIR,
                              save_intermediate_results=SIR, x0=x0, max_x=max_x, c1=c1, c2=c2)
    Extra_Tasks = mop.ExtraTasks(func=func, a=a, b=b, eps=eps, max_iter=max_iter, print_intermediate_results=False,
                                 save_intermediate_results=True, method=task, x0=x0, max_x=max_x, c1=c1, c2=c2)
    funcs = {
        'Метод золотого сечения': Example.gss,
        'Метод парабол': Example.quadratic_approximation,
        'Метод Брента': Example.brent,
        'Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно': Example.BFGS}

    if st.sidebar.button('Click'):
        st.header('Задание 1. ' + task)
        st.dataframe(funcs[task]().iloc[:, [0, 1, 3]])

        st.header('Задание 3. График работы методы')
        st.plotly_chart(Extra_Tasks.q3())

        st.header('Задание 4. График сходимости алгоритма')
        st.plotly_chart(Extra_Tasks.q4())

        st.header('Задание 5. Сравнение производительности алгоритмов')
        st.dataframe(Extra_Tasks.q5())


def page3():
    st.title('Многомерная (до 4-х переменных) оптимизация с помощью градиентных методов')
    st.write("***")

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Ввести нужную функцию в стиле numpy в поле “Введите целевую функцию”.
        Можно использовать функции, реализованные в numpy: np.sin(x1) + np.exp(x2)""")
        st.write("""2. Выбрать нужный метод""")
        st.write(
            """3. Ввести список координат начальной точки. Размер вектора начальной точки зависит от кол-ва переменных""")
        st.write("""4. Ввести максимально возможное количество итераций для цикла.""")
        st.write("""5. Ввести скорость обучения.""")
        st.write("""6. Выбрать, нужно ли выводить промежуточные результаты""")
        st.write("""7. Ввести дополнительные параметры""")
        st.write("""8. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация графиков')
        st.write("""1. На первом графике изображен "путь" функции с первой итерации до последней""")
        st.write("""2. На втором графике построены графики сходимости.""")

    st.sidebar.header('Ввод данных')
    func = st.sidebar.text_input('Целевая функция', value='x1**2 - 2*x1*x2 + x2**2')
    method = st.sidebar.selectbox('Выберите метод', options=[
        'Градиентный спуск с постоянным шагом',
        'Градиентный спуск с дроблением шага',
        'Метод наискорейшего спуска',
        'Метод сопряженных градиентов'])
    x0 = eval(
        st.sidebar.text_input('Список координат начальной точки. Пример для 3-х переменных: [2, 5, 1]', value='[0,0]'))

    str_value = 'lambda x1'
    for i in range(2, len(x0) + 1):
        str_value += f', x{i}'
    str_value += ':'
    func = eval(str_value + func)

    max_iterations = st.sidebar.slider('Макс. кол-во итераций, ', min_value=1, max_value=1000, value=500, step=25)
    if method != 'Метод сопряженных градиентов':
        lr = st.sidebar.number_input("learning rate", value=1e-3, min_value=1e-5, max_value=1., step=1e-6,
                                     format='%.6f')
        PIR = st.sidebar.selectbox(label='Вывод промежуточных результатов', options=[False, True])
    # func, n_variables, x0, max_iterations, eps, SIR, PIR, lr

    if st.sidebar.button('Найти минимум'):
        if method != 'Метод сопряженных градиентов':
            obj = gradient_methods.GradientMethod(func=func, x0=x0, max_iterations=max_iterations, PIR=PIR, lr=lr)
            result = obj.minimize2(method)
            st.write(result)
            st.plotly_chart(obj.visualize_contour())
            st.plotly_chart(obj.visualize_convergence())
        else:
            result = gradient_methods.GradientMethod(func=func, x0=x0, max_iterations=max_iterations).minimize2()
            st.write(result)


def page4():
    st.title('Регрессия')
    st.write('Приложение строит модель регрессии, выводит коэффиценты и строит график')
    st.write('***')

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Выбрать тип регрессии""")
        st.write("""2. Ввести массив предикаторов X.""")
        st.write("""3. Ввести вектор предсказываемой переменной (y).""")
        st.write("""4. Выбрать тип регуляризации""")
        st.write("""5. Выбрать, нужно ли построение графика, только для до 2-х признаков""")
        st.write("""6. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация графика')
        st.write(
            """1. Если кол-во признаков = 1, то будет построены график рассеивания исходных данных и линия - результат работы модели""")
        st.write(
            """2. Если кол-во признаков = 2, то будет построены график рассеивания исходных данных и плоскость - результат работы модели""")

    st.sidebar.header('Ввод данных')
    chosen_model = st.sidebar.selectbox('Выберите тип регрессии', options=[
        'Классическая',
        'Полиномиальная',
        'Экспоненциальная'
    ])
    if chosen_model == 'Полиномиальная':
        poly_degree = st.sidebar.slider('Выберите максимальную степерь полиномиальной модели', 1, 10)
    else:
        poly_degree = None
    X = st.sidebar.text_input('Введите массив предикаторов X.',
                              help='Пример для ввода данных для 3-х регрессоров: [[1,0,3], [2,3,1], [4,2,9], [8,4,9]]')

    Y = st.sidebar.text_input('Введите вектор предсказываемой переменной (y).', help='Пример: [1, 2, 3, 4]')

    reg = st.sidebar.selectbox('Вид регуляризации', options=[None, 'l1', 'l2'])
    if reg is not None:
        alpha = st.sidebar.number_input('Вес "штрафа" регуляризации')
    else:
        alpha = None

    plot = st.sidebar.selectbox('Построение графика', options=[False, True])

    model_dict = {'Классическая': 'classic',
                  'Полиномиальная': 'poly',
                  'Экспоненциальная': 'expo'}

    if st.sidebar.button('Click'):
        st.header('Введенные данные')
        st.write({
            'Тип регрессии': chosen_model,
            'X': X,
            'Y': Y,
            'Макс. степень полиномиальной модели': poly_degree,
            'Регуляризация': reg,
            'Вес штрафа': alpha})
        st.write('***')
        st.header('Результат программы')
        model = LinearRegression.LinearModels(model=model_dict[chosen_model])
        model.fit(X=eval(X), Y=eval(Y), poly_degree=poly_degree, alpha=alpha)
        st.write(model.analytical_func())
        weights = pd.DataFrame(
            {'Полученные веса': model.get_weights()},
            index=[f'w{i}' for i in range(len(model.get_weights()))])
        st.dataframe(weights)

        if plot:
            st.plotly_chart(model.visualize())


def page6():
    st.title('Классификация')
    st.write(
        'Приложение решает задачу бинарной классификации с помощью линейного классификатора, выводит результаты и график (только для 2-х признаков)')
    st.write('***')

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Ввести/загрузить данные""")
        st.write("""2. Выбрать тип модели""")
        st.write("""3. Выбрать, нужно ли рименение гауссовской радиально-базисной функции""")
        st.write("""4. Выбрать тип регуляризации""")
        st.write("""5. Ввести кол-во эпох(циклов)""")
        st.write("""7. Ввести кол-во батчей(подциклов)""")
        st.write("""8. Ввести размер батчей(подциклов)""")
        st.write("""9. Ввести скорость обучения модели""")
        st.write("""10. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация вывода')
        st.write("""1. На экран отобразятся специальные отчеты (classification_report из библиотеки scipy), 
        которые включают в себя различные метрики для оценивания работы модели. 
        см. больше информации о метриках: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support""")
        st.write("""2. Если кол-во признаков = 2, то будет построены график рассеивания исходных данных и
        разделительная черта  - результат работы модели. (график можно интерпретировать так: если разделительная черта - вертикальная прямая
        , то точки слева от прямой принадлежат к классу -1, иначе к классу 1.)""")

    st.sidebar.header('Ввод данных')
    X, y = [0], [0]
    chosen_input_format = st.sidebar.selectbox('Выберите способ ввода данных', options=[
        'Выбрать игрушечный (2 признака) датасет',
        'Загрузить файл формата csv',
        'Ввести самоcтоятельно данные'
    ])
    if chosen_input_format == 'Выбрать игрушечный (2 признака) датасет':
        chosen_dataset = st.sidebar.selectbox('Выберите датасет', options=[
            'Линейно разделимые классы', 'Линейно не разделимые классы'])

        if chosen_dataset == 'Линейно разделимые классы':
            X, y = make_blobs(n_samples=1000, centers=2, n_features=2, center_box=(-10, 10), random_state=1)

        else:
            X, y = make_gaussian_quantiles(n_classes=2, n_samples=1000, random_state=1)

    elif chosen_input_format == 'Загрузить файл формата csv':
        uploaded_data = st.sidebar.file_uploader('Выберите файл формата csv, столбец меток должен быть последним')
        df = pd.read_csv(uploaded_data)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    elif chosen_input_format == 'Ввести самоcтоятельно данные':
        X = st.sidebar.text_input('Введите массив X.',
                                  help='Пример для ввода данных для 3-х признаков: [[1,0], [2,3], [4,2], [8,4]]')
        y = st.sidebar.text_input('Введите вектор меток y.', help='Пример: [1,0,1,0]')
        X = eval(X)
        y = eval(y)

    X = torch.tensor(X, dtype=torch.float)
    y[y == 0] = -1
    y = torch.tensor(y, dtype=torch.float)

    st.sidebar.header('Параметры модели')
    chosen_model = st.sidebar.selectbox('Выберите тип модели', options=[
        'Логистическая регрессия',
        'Метод опорных векторов',
    ])
    if chosen_model == 'Логистическая регрессия':
        svm = False
    else:
        svm = True

    rbf = st.sidebar.selectbox('Применение гауссовской радиально-базисной функции', options=[
        False,
        True
    ])

    if rbf:
        gamma = st.sidebar.number_input("Множитель радиально-базисной функции",
                                        value=1e-3, min_value=1e-5, max_value=1., step=1e-6, format='%.6f')
    else:
        gamma = None

    reg = st.sidebar.selectbox('Тип регуляризации', options=[
        None,
        'l1',
        'l2'
    ])

    if reg is not None:
        alpha = st.sidebar.number_input('Вес "штрафа" регуляризации')
    else:
        alpha = None

    num_epo = st.sidebar.number_input('Кол-во эпох', value=10)
    num_batch = st.sidebar.number_input('Кол-во батчей в эпохе', value=50)
    batch_size = st.sidebar.number_input('Размер батчей', value=10)
    lr = st.sidebar.number_input("learning rate", value=1e-3, min_value=1e-5, max_value=1., step=1e-6, format='%.6f')

    if st.sidebar.button('Click'):
        model = Classifier.Classifier()
        train_result, test_result = model.fit(X, y, num_epo, lr, num_batch, batch_size, reg, alpha, gamma, svm)

        st.caption(
            'см. больше информации о метриках снизу: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support')
        st.text('Результаты модели на обучающей выборке:\n ' + train_result)
        st.write('***')
        st.text('Результаты модели на тестовой выборке:\n ' + test_result)
        st.write('***')
        st.plotly_chart(model.plot())


def page5():
    st.title('Метод внутренней точки')
    st.write('Приложение решает задачу оптимизации с ограничениями')
    st.write('***')

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write("""Примечания: многочлены нужно разделять пробелами и коэффиценты следует писать перед переменными
        : верно - 2*x1 - 3*x2, не верно - x1*3-x2*2""")
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Ввести нужную функцию в стиле numpy в поле “Введите целевую функцию”.
        Можно использовать функции, реализованные в numpy: np.sin(x1) + np.exp(x2)""")
        st.write("""2. Выбрать нужный метод""")
        st.write("""3. Ввести ограничения в виде списка: [2*x1 + 5*x2 <= 0, 3*x1 + 4*x2 -5 <= 0]""")
        st.write("""4. Ввести начальную точку в виде списка""")
        st.write("""5. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация графиков')
        st.write("""1. На экране отобразятся оптимальная точка и значение функции в ней""")
        st.write("""2. Если кол-во переменных = 2, будет построен 3d график, на котором изображены целевая функция,
         ограничения, нулевая плоскость и точка. Для ограничения вида равенств: Точка должна находиться на пересечении трех плоскостей.
         Второй график является интерпретацией первого в меньшей размерности""")

    st.sidebar.header('Ввод данных')

    func = st.sidebar.text_input('Целевая функция', value='2*x1 + 3*x2')

    method = st.sidebar.selectbox('Выберите метод', options=[
        'Метод Ньютона',
        'Метод логарифмических барьеров',
        'Прямо-двойственный метод внутренней точки'])

    if method == 'Метод Ньютона':
        limit_desc = 'Ограничения типа равенств'
        value = '2*x1 - x2 - 10, x1 + 4*x2 - 13'
    else:
        limit_desc = 'Ограничения типа равенств и неравенств'
        value = '2*x1 - x2 - 10, x1 + 4*x2 - 13'
    limits = st.sidebar.text_input(
        limit_desc,
        value=value
    )
    limits = limits.split(',')

    x0 = st.sidebar.text_input(
        'Начальная точка',
        value='[0, 0]'
    )
    x0 = eval(x0)

    if method != 'Метод Ньютона':
        tol = st.sidebar.number_input(
            "Точность",
            value=1e-3,
            step=1e-6,
            format='%.6f')
    if method == 'Метод логарифмических барьеров':
        max_steps = st.sidebar.number_input(
            'Макс. кол-во итераций',
            value=50
        )

    function_dict = {
        'Метод Ньютона': tz5.Newton,
        'Метод логарифмических барьеров': tz5.logBarMethod,
        'Прямо-двойственный метод внутренней точки': tz5.inequality
    }

    if method == 'Метод Ньютона':
        result = tz5.Newton(
            func=func,
            equality=limits,
            x0=x0
        )

    elif method == 'Метод логарифмических барьеров':
        result = tz5.logBarMethod(
            func=func,
            restrictions=limits,
            start_point=x0,
            max_steps=max_steps,
            tol=tol
        )

    else:
        result = tz5.inequality(
            func=func,
            us=limits,
            x0=x0,
            tol=tol
        )

    if st.sidebar.button('Click'):
        st.header('Полученный результат')
        st.write({
            'X': result[0],
            'f(X)': result[1]
        })
        st.plotly_chart(tz5.visualize3d(func, limits, result))
        st.plotly_chart(tz5.visualize(func, limits, result))


def page7():
    st.title('Метод секущих плоскостей')
    st.write('Приложение находит положительное целочисленное решение задачи линейного программирования')
    st.write('***')

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Выбрать нужный метод""")
        st.write("""2. Выбрать тип задачи""")
        st.write("""3. Задать коэффициенты целевой функциив в виде списка: [2,3] -> 2x1 + 3x2""")
        st.write(
            """4. Задать коэффициенты ограничений в виде массива: [[2,1], [1,4]] -> 2x1+x2 <= b1, x1 + 4x2 <= b2""")
        st.write(
            """5. Задать коэффициенты свободных членов ограничений в виде списка: [10, 13] -> g1 <= 10, g2 <= 13""")
        st.write("""6. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация графиков')
        st.write("""1. На экране отобразятся оптимальная точка и значение функции в ней""")
        st.write(
            """2. Если кол-во переменных = 2, будет построен графики ограничений, оптимальная точка и контурный график целевой функции""")

    st.sidebar.header('Ввод данных')

    method = st.sidebar.selectbox('Метод', ['Метод Гомори', 'Метод ветвей и границ'])
    mode = st.sidebar.selectbox("Тип задачи", ['MAX', 'MIN'])
    method_dict = {
        'Метод Гомори': gamori.main,
        'Метод ветвей и границ': b_and_b.bb
    }
    c = st.sidebar.text_input(
        'Коэффициенты заданной функции',
        help='Пример: [2,3] <-> 2x1 + 3x2',
        value='[2,3]')
    a = st.sidebar.text_input(
        'Коэффициенты заданных ограничений',
        help='Пример: [[2, 1],[1, 4]] <-> g1 = 2x1 + x2 <= b1, g2 = x1 + 4x2 <= b2',
        value='[[2,1], [1,4]]'
    )
    b = st.sidebar.text_input(
        'Коэффициенты свободных членов ограничений',
        help='Пример: [10, 13] <-> b1 = 10, b2 = 13',
        value='[10, 13]')

    c, a, b = [array(eval(list)) for list in [c, a, b]]

    if st.sidebar.button('Click'):
        result = method_dict[method](c, a, b, mode)
        st.write(result)
        st.plotly_chart(gamori.visualize(c, a, b, result))


def page8():
    st.title('Методы стохастической оптимизации')
    st.write('Приложение находит решает задачу оптимизации различными стохастическими методами')
    st.write('***')

    if st.checkbox('Показать инструкцию', value=True):
        st.header('Ввод данных')
        st.write('Для работы приложения нужно ввести входные данные слева в боковой панели')
        st.write("""1. Выбрать нужный метод""")
        st.write("""2. Ввести целевую функцию в явном виде. *Вместо x1, x2, ... xn требуется использовать
        x[0], x[1] и x[n-1]*""")
        st.write("""3. Ввести остальные параметры""")
        st.write("""4. Нажать кнопку Click на боковой панели, результаты программы отобразятся ниже документации""")

        st.header('Интерпретация результатов')
        st.write("""1. На экране отобразятся найденная точка и значение функции в ней""")
        st.write("""2. Будет построен график изменения значений функций каждую итерацию""")
    st.sidebar.header('Ввод данных')

    method = st.sidebar.selectbox('Метод', ['Метод имитации отжига', 'Генетического алгоритм'])

    func = st.sidebar.text_input(
        'Целевая функция',
        value='x[0]**2.0 + (x[1]-5)**2',
        help='Объявляйте переменные как элементы списка: x1 -> x[0], x2 -> x[1]')

    def objective(x):
        return eval(func)

    n_iterations = st.sidebar.number_input(
        'Кол-во итераций',
        value=10
    )

    if method == 'Метод имитации отжига':
        start_point = st.sidebar.text_input(
            'Начальная точка',
            value='[1,4]'
        )

        step_size = st.sidebar.number_input(
            'Размер шага',
            value=0.1
        )

        temp = st.sidebar.number_input(
            '   Начальная температура',
            value=10
        )

        best, score, scores = tz8.simulated_annealing(objective, eval(start_point), n_iterations, step_size, temp)

    elif method == 'Генетического алгоритм':
        bounds = eval(st.sidebar.text_input(
            'Области нахождения минимума',
            value='[[-5, 5], [-5,5]]'
        ))

        n_pop = st.sidebar.number_input(
            'Кол-во популяций, должно быть кратно 10',
            value=100,
            step=10
        )

        best, score, scores = tz8.genetic_algorithm(objective, bounds, 16, n_iterations, n_pop, 0.9,
                                                    1.0 / (float(16) * len(bounds)))
        best = tz8.decode(bounds, 16, best)
    if st.sidebar.button('Click'):
        st.write('f(%s) = %f' % (best, score))
        st.plotly_chart(tz8.visualize(scores))
