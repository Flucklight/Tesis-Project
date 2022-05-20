from numpy.random import rand
from numpy.random import choice
from numpy.random import uniform
from numpy import clip
from numpy import asarray
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
    return Individual(asarray(data))


# evaluate a individual
def evaluate(ind, objective):
    ind.score = objective(ind.gen)


# get score of individual
def get_score(ind):
    return ind.score


# define mutation operation
def mutation(x, r_mut):
    return x[0].gen + r_mut * (x[1].gen - x[2].gen)


# define boundary check operation
def check_bounds(mutated, bounds):
    return asarray([clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))])


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target.gen[i] for i in range(dims)]
    return Individual(asarray(trial))


def differential_evolution(objective, bounds, n_iter, n_pop, r_mut, r_cross, heterogeneity, num_test):
    # initialise population of candidate solutions randomly within the specified bounds
    population = [generate(bounds) for _ in range(n_pop)]
    # evaluate initial population of candidate solutions
    for ind in population:
        evaluate(ind, objective)
    # find the best performing vector
    population.sort(key=get_score)
    # find the best performing vector of initial population
    best, best_eval = population[0].gen.tolist(), population[0].score
    # initialise list to store the objective function value at each iteration
    table = DataFrame()
    # store the children
    children = list()
    # run iterations of the algorithm
    for gen in range(n_iter):
        children.clear()
        selected = population.copy()
        # iterate over all candidate solutions
        for ind in selected:
            # choose three candidates, a, b and c, that are not the current one
            a, b, c = choice(selected, 3, replace=False)
            # perform mutation
            mutated = mutation([a, b, c], r_mut)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, ind, len(bounds), r_cross)
            # compute objective function value for trial vector
            evaluate(trial, objective)
            # perform selection
            children.append(trial)
        # select best children
        children.sort(key=get_score)
        # replace population
        population.clear()
        # use heterogeneity to append a percentage of the best
        for i in range(int(ceil(n_pop * heterogeneity))):
            if selected[0].score <= children[0].score:
                population.append(selected.pop(0))
            else:
                population.append(children.pop(0))
        # and other to append a percentage of random
        for i in range(int(floor(n_pop * (1 - heterogeneity)))):
            population.append(choice(children, replace=False))
        # order by the best
        population.sort(key=get_score)
        # store info to graph
        dic = {'Algorithm': ['differential_evolution'], 'Objective': [objective.__name__.capitalize()], 'Test': [num_test], 'Heterogeneity': [heterogeneity], 'Generation': [gen], 'Result': [population[0].score]}
        table = concat([table, DataFrame(dic)], ignore_index=True)
        # check for new best solution
        if population[0].score < best_eval:
            best, best_eval = population[0].gen.tolist(), population[0].score
            print('\tGeneration {} -> new best {} = {}'.format(gen, best, best_eval))
    return [best, best_eval, table]
