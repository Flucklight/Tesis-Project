from numpy.random import randn
from numpy.random import uniform
from numpy.random import choice
from numpy import ceil
from numpy import floor
from pandas import DataFrame
from pandas import concat
from objects.individual import Individual


# generate values of population
def generate(bounds):
    data = list()
    for s in bounds:
        data.append(uniform(s[0], s[1]))
    return Individual(data)


# evaluate a individual
def evaluate(ind, objective):
    ind.score = objective(ind.gen)


# get score of individual
def get_score(ind):
    return ind.score


# check if a point is within the bounds of the search
def in_bounds(point, bounds):
    # enumerate all dimensions of the point
    for d in range(len(bounds)):
        # check if out of bounds for this dimension
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True


# evolution strategy (mu, lambda) algorithm
def evolutionary_strategies(objective, bounds, n_iter, step_size, mu, lam, heterogeneity, num_test):
    # store information
    data = list()
    table = DataFrame()
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = [generate(bounds) for _ in range(lam)]
    # evaluate fitness for the population
    for ind in population:
        evaluate(ind, objective)
    # rank individuals in ascending order
    population.sort(key=get_score)
    # trackers for the best solutions
    best, best_eval = population[0].gen, population[0].score
    # perform the search
    for gen in range(n_iter):
        # create children from parents
        children = list()
        for ind in population[:mu]:
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = ind.gen + randn(len(bounds)) * step_size
                children.append(Individual(child))
        # evaluate fitness for the children
        for ind in children:
            evaluate(ind, objective)
        # rank children in ascending order
        children.sort(key=get_score)
        # copy the population to be selected
        selected = population.copy()
        # replace population
        population.clear()
        # use heterogeneity to append a percentage of the best
        for i in range(int(ceil(lam * heterogeneity))):
            if selected[0].score <= children[0].score:
                population.append(selected.pop(0))
            else:
                population.append(children.pop(0))
        # and other to append a percentage of random
        for i in range(int(floor(lam * (1 - heterogeneity)))):
            population.append(choice(children, replace=False))
        # order by the best
        population.sort(key=get_score)
        # store info to graph
        data.append(population[0].score)
        dic = {'Algorithm': ['evolutionary_strategies'], 'Objective': [objective.__name__.capitalize()], 'Test': [num_test], 'Heterogeneity': [heterogeneity], 'Generation': [gen], 'Result': [population[0].score]}
        table = concat([table, DataFrame(dic)], ignore_index=True)
        # check if this parent is the best solution ever seen
        if population[0].score < best_eval:
            best, best_eval = population[0].gen, population[0].score
            print('Generation {} -> new best {} = {}'.format(gen, best, best_eval))
    return [best, best_eval, data, table]
