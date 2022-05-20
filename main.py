import os

from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import sin
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
def ackley(values):
    x, y = 0.0, 0.0
    for value in values:
        x += value ** 2
        y += cos(2 * pi * value)
    return -20.0 * exp(-0.2 * sqrt(x / len(values))) - exp(y / len(values)) + e + 20


def levy(values):
    w1 = 1 + ((values[0] - 1) / 4)
    wd = 1 + ((values[-1] - 1) / 4)
    term1 = sin(pi * w1) ** 2
    term3 = ((wd - 1) ** 2) * (1 + (sin(2 * pi * wd) ** 2))
    term2 = 0.0
    for value in values:
        w = 1 + ((value - 1) / 4)
        term2 += ((w - 1) ** 2) * (1 + 10 * (sin(pi * w + 1) ** 2))
    return term1 + term2 + term3


def sphere(values):
    x = 0.0
    for value in values:
        x += value ** 2
    return x


def rosenbrock(values):
    value = 0.0
    for i in range(len(values) - 1):
        value += 100 * ((values[i + 1] - values[i] ** 2) ** 2) + ((values[i] - 1) ** 2)
    return value


def zakharov(values):
    x, y = 0.0, 0.0
    for i in range(len(values)):
        x += values[i] ** 2
        y += 0.5 * i * values[i]
    return x + (y ** 2) + (y ** 4)


OBJECTIVES = [ackley, levy, sphere, rosenbrock, zakharov]
METHODS = [genetic_algorithm, evolutionary_strategies, differential_evolution]


def get_bounds(function, variables):
    bounds = list()
    if function.__name__ == 'ackley':
        bounds = asarray([[-32.768, 32.768] for _ in range(variables)])
    elif function.__name__ == 'levy':
        bounds = asarray([[-10.0, 10.0] for _ in range(variables)])
    elif function.__name__ == 'sphere':
        bounds = asarray([[-5.12, 5.12] for _ in range(variables)])
    elif function.__name__ == 'rosenbrock':
        bounds = asarray([[-5.0, 10.0] for _ in range(variables)])
    elif function.__name__ == 'zakharov':
        bounds = asarray([[-5.0, 10.0] for _ in range(variables)])
    return bounds


def run(heterogeneity, method, function, n_vars, n_test, show=True):
    # init variables
    best = []
    score = 0.0
    objective = function
    # init store variables
    table = DataFrame()
    info = DataFrame()
    # define range for input
    bounds = get_bounds(function, n_vars)
    # define the total iterations
    n_iter = 1000
    # define the population size
    n_pop = 100
    for test in range(n_test):
        # the heterogeneity parameter bound in 0 to 1, 0 represent only random subjects and 1 only the best subjects
        print('Method {} Function {} Heterogeneity {} Test {}'.format(method.__name__, objective.__name__, heterogeneity, test))
        if method.__name__ == 'genetic_algorithm':
            # perform the genetic algorithm search
            best, score, table = method(objective, bounds, n_iter, n_pop, r_cross=0.8, r_mut=0.1, alpha=0.25, heterogeneity=heterogeneity, num_test=test)
        elif method.__name__ == 'evolutionary_strategies':
            # perform the evolution strategy search
            best, score, table = method(objective, bounds, n_iter, step_size=0.15, mu=100, lam=100, heterogeneity=heterogeneity, num_test=test)
        if method.__name__ == 'differential_evolution':
            # perform differential evolution search
            best, score, table = method(objective, bounds, n_iter, n_pop, r_cross=0.8, r_mut=0.1, heterogeneity=heterogeneity, num_test=test)
        # show results
        print('Soluction: {} = {}\n'.format(best, score))
        # create the directories to save the data
        if not os.path.exists('./results/{}/{}/Heterogeneity-{}'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100))):
            os.makedirs('./results/{}/{}/Heterogeneity-{}/Plots'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100)))
            os.makedirs('./results/{}/{}/Heterogeneity-{}/Tables'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100)))
        # graph data
        table.plot(x='Generation', y='Result')
        pyplot.title('{} - {} - Heterogeneity {} - Test {}'.format(method.__name__.capitalize(), objective.__name__.capitalize(), heterogeneity, test))
        pyplot.xlabel('Generation')
        pyplot.ylabel('Best Individual Score')
        pyplot.savefig('./results/{}/{}/Heterogeneity-{}/Plots/Test-{}.png'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100), test))
        if show:
            pyplot.show()
        # Append the results
        table.to_csv('./results/{}/{}/Heterogeneity-{}/Tables/Table-Test-{}.csv'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100), test))
        info = concat([info, table], ignore_index=True)
    info.to_csv('./results/{}/{}/Heterogeneity-{}/Result-Table.csv'.format(method.__name__.capitalize(), objective.__name__.capitalize(), round(heterogeneity * 100)))
    return info


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    result = DataFrame()
    t = 20
    v = 10
    for m in METHODS:
        for f in OBJECTIVES:
            h = 0.0
            while h <= 1.0:
                r = run(h, m, f, v, t, show=False)
                result = concat([result, r], ignore_index=True)
                h += 0.25
    result.to_csv('./results/Result-Table.csv')
    print('Done!')
