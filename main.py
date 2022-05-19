import os

from numpy import exp
from numpy import sqrt
from numpy import power
from numpy import cos
from numpy import sin
from numpy import abs
from numpy import e
from numpy import pi
from numpy import asarray

from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat

from algorithms.ga import genetic_algorithm
from algorithms.es import evolutionary_strategies
from algorithms.de import differential_evolution


# objective functions
def ackley(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20


def eggholder(v):
    x, y = v
    return -(y + 47.0) * sin(sqrt(abs((x / 2) + (y + 47)))) - x * sin(sqrt(abs(x - (y + 47))))


def himmelblau(v):
    x, y = v
    return power((power(x, 2) + y + 11), 2) + power((x + power(y, 2) - 7), 2)


def rosenbrock(v):
    x, y = v
    return 100 * power((y - power(x, 2)), 2) + power((1 - x), 2)


def beale(v):
    x, y = v
    return 100 * power((1.5 - x + x * y), 2) + power((2.25 - x + x * power(y, 2)), 2) + power((2.625 - x + x * power(y, 3)), 2)


OBJECTIVES = [ackley, eggholder, himmelblau, rosenbrock, beale]
METHODS = [genetic_algorithm, evolutionary_strategies, differential_evolution]


def get_bounds(function):
    bounds = list()
    if function.__name__ == 'ackley':
        bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
    elif function.__name__ == 'eggholder':
        bounds = asarray([[-512.0, 512.0], [-512.0, 512.0]])
    elif function.__name__ == 'himmelblau':
        bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
    elif function.__name__ == 'rosenbrock':
        bounds = asarray([[-1000.0, 1000.0], [-1000.0, 1000.0]])
    elif function.__name__ == 'beale':
        bounds = asarray([[-4.5, 4.5], [-4.5, 4.5]])
    return bounds


def run(heterogeneity, method, function, n_test, show=True):
    # init variables
    best = []
    score = 0.0
    objective = function
    # init store variables
    data = []
    table = DataFrame()
    info = DataFrame()
    # define range for input
    bounds = get_bounds(function)
    # define the total iterations
    n_iter = 1000
    # define the population size
    n_pop = 100
    for test in range(n_test):
        # the heterogeneity parameter bound in 0 to 1, 0 represent only random subjects and 1 only the best subjects
        print('Heterogeneity {} Test {} Function {} Method {}'.format(heterogeneity, test, objective.__name__, method.__name__))
        if method.__name__ == 'genetic_algorithm':
            # perform the genetic algorithm search
            best, score, data, table = method(objective, bounds, n_iter, n_pop, r_cross=0.8, r_mut=0.1, alpha=0.25, heterogeneity=heterogeneity, num_test=test)
        elif method.__name__ == 'evolutionary_strategies':
            # perform the evolution strategy search
            best, score, data, table = method(objective, bounds, n_iter, step_size=0.15, mu=100, lam=100, heterogeneity=heterogeneity, num_test=test)
        if method.__name__ == 'differential_evolution':
            # perform differential evolution search
            best, score, data, table = method(objective, bounds, n_iter, n_pop, r_cross=0.8, r_mut=0.1, heterogeneity=heterogeneity, num_test=test)
        # show results
        print('Soluction: {} = {}\n'.format(best, score))
        # create the directories to save the data
        if not os.path.exists('./results/{}/{}/Heterogeneity-{}'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100))):
            os.makedirs('./results/{}/{}/Heterogeneity-{}'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100)))
        # graph data
        if show:
            table.plot(x='Generation', y='Result')
            pyplot.title('{} - {} - Heterogeneity {} - Test {}'.format(method.__name__.capitalize(), objective.__name__.capitalize(), heterogeneity, test))
            pyplot.xlabel('Generation')
            pyplot.ylabel('Best Individual Score')
            pyplot.savefig('./results/{}/{}/Heterogeneity-{}/Test-{}.png'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100), test))
            pyplot.show()
        # Append the results
        table.to_csv('./results/{}/{}/Heterogeneity-{}/Table-Test-{}.csv'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100), test))
        info = concat([info, table], ignore_index=True)
    info.to_csv('./results/{}/{}/Heterogeneity-{}/Result-Table.csv'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100)))
    return info


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    result = DataFrame()
    t = 20
    for m in METHODS:
        for f in OBJECTIVES:
            h = 0.0
            while h <= 1.0:
                r = run(h, m, f, t, show=True)
                result = concat([result, r], ignore_index=True)
                h += 0.25
    result.to_csv('./results/Result-Table.csv')
    print('Done!')
