from numpy.random import uniform
from numpy.random import rand
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


# elitism selection
def selection(pop, scores):
    # first select ind
    selection_ix = 0
    for ix in range(len(pop)):
        # check if better
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    selected = pop.pop(selection_ix)
    return selected


# crossover two parents to create two children
def crossover(p1, p2, r_cross, alpha):
    # children are copies of parents by default
    c1, c2 = p1.gen.copy(), p2.gen.copy()
    # check for recombination
    if rand() < r_cross:
        # perform crossover
        c1 = [p1.gen[0] - alpha * (p2.gen[0] - p1.gen[1]), p1.gen[1] - alpha * (p2.gen[1] - p1.gen[1])]
        c2 = [p2.gen[0] - alpha * (p1.gen[0] - p2.gen[1]), p2.gen[1] - alpha * (p1.gen[1] - p2.gen[1])]
    return [Individual(c1), Individual(c2)]


# mutation operator
def mutation(c, bounds, r_mut):
    for v in range(len(c.gen)):
        # check for a mutation
        if rand() < r_mut:
            # change the value
            c.gen[v] = uniform(bounds[v][0], bounds[v][1])


# genetic algorithm
def genetic_algorithm(objective, bounds, n_iter, n_pop, r_cross, r_mut, alpha, heterogeneity, num_test):
    # store information
    data = list()
    table = DataFrame()
    # initial population of random values
    population = [generate(bounds) for _ in range(n_pop)]
    # evaluate all candidates in the population
    for ind in population:
        evaluate(ind, objective)
    population.sort(key=get_score)
    # keep track of best solution
    best, best_eval = population[0].gen, population[0].score
    # enumerate generations
    for gen in range(n_iter):
        # select parents
        selected = population.copy()
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross, alpha):
                # mutation
                mutation(c, bounds, r_mut)
                # evaluate the candidate
                evaluate(c, objective)
                # store for next generation
                children.append(c)
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
        data.append(population[0].score)
        dic = {'Algorithm': ['genetic_algorithm'], 'Objective': [objective.__name__.capitalize()], 'Test': [num_test], 'Heterogeneity': [heterogeneity], 'Generation': [gen], 'Result': [population[0].score]}
        table = concat([table, DataFrame(dic)])
        # check for new best solution
        if population[0].score < best_eval:
            best, best_eval = population[0].gen, population[0].score
            print('Generation {} -> new best {} = {}'.format(gen, best, best_eval))
    return [best, best_eval, data, table]
