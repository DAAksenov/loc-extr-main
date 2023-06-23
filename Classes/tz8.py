from numpy import asarray
from numpy import exp
from numpy.random import randn, rand, seed, randint
import numpy as np
import plotly.graph_objects as go
from Classes.Classifier import Classifier

#Задание2
def pegasos(X, y, num_epo, lr, num_batch, batch_size, reg, alpha, gamma):
    """
    Решение задачи классификации методом опорных векторов,
    минимизация функции потерь с помощью стохастического градиентного спуска

    Parameteres:
        X: torch.Tensor - Матрица признаков;
        y: torch.Tensor - Вектор меток классов;
        num_epo: int - Кол-во эпох;
        lr: float - скорость обучения;
        num_batch: Кол-во батчей/подвыборок
        batch_size: Размер батчей;
        reg: str - Тип регуляризации;
        alpha: Коэффицент регуляризации;
        gamma: float - Коэффицент rbf, если None, то rbf не используется
    """
    model = Classifier.Classifier()
    train_result, test_result = model.fit(X, y, num_epo, lr, num_batch, batch_size, reg, alpha, gamma, svm = True)
    return train_result, test_result

def simulated_annealing(objective, start_point, n_iterations, step_size, temp):
    """
    Реализация метода имитации отжига

    Parameters:
        objective: function - целевая функция;
        start_point: list - начальная точка;
        n_iterations: int - макс. кол-во итераций;
        step_size: float - Размер шага;
        temp: float - начальная температура;
    """
    best = start_point
	# evaluate the initial point
    best_eval = objective(best)
	# current working solution
    curr, curr_eval = best, best_eval
    scores = list()
	# run the algorithm
    for i in range(n_iterations):
		# take a step
        candidate = curr + randn(len(start_point)) * step_size
		# evaluate candidate point
        candidate_eval = objective(candidate)
		# check for new best solution
        if candidate_eval < best_eval:
			# store new best point
            best, best_eval = candidate, candidate_eval
			# keep track of scores
            scores.append(best_eval)
			# report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
		# difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
        t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
		# check if we should keep the new point
        if diff < 0 or rand() < metropolis:
			# store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval, scores]


def visualize(scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y = scores, name = 'F(X) per iterations'))
    return fig



 
def objective(x):
	return x[0]**2.0
 
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    """
    Двоичный код -> десятичное число
    """
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
		# extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
		# convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
		# convert string to integer
        integer = int(chars, 2)
		# scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
        decoded.append(value)
    return decoded
 
# tournament selection
def selection(pop, scores, k=3):
    """
    Отбор кандидатов
    """
	# first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 
# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]
 
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    """
    Реализация метода имитации отжига

    Parameters:
        objective: function - целевая функция;
        bounds: array - области нахождения минимума;
        n_bits: int - размерность двоичного числа (каждый предок и потомок описываются двочиным числом)
        n_iter: int - макс. кол-во итераций;
        n_pop: int - кол-во популяций;
    """
    results = []

    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    for gen in range(n_iter):
        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = [objective(d) for d in decoded]
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
        results.append(best_eval)
        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        pop = children
    return [best, best_eval, results]
