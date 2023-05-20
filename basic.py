import random

import numpy as np
import matplotlib.pyplot as plt

LOW = -1
HIGH = 2
MUTATION_P = 0.5
POPSIZE = 10


def genotypeToFenotype(element):
    f = 0
    for i, j in zip(element, range(11, -1, -1)):
        f += i * 2 ** j
    return (f / 2 ** 12) * (HIGH - LOW) + LOW


def getFitness(x):
    return x * np.sin(10 * np.pi * x) + 2


def represent():
    population = np.random.randint(0, 2, (POPSIZE, 12))
    fenotype = np.empty(POPSIZE, np.float64)
    for i in range(POPSIZE):
        fenotype[i] = genotypeToFenotype(population[i])

    fitness = getFitness(fenotype)
    p = fitness / fitness.sum()
    return population, fenotype, p


def crossover(population: list):
    out = []
    while len(population) > 0:
        a1 = random.choice(population)
        population.remove(a1)
        a2 = random.choice(population)
        population.remove(a2)

        border = random.randint(0, 11)
        a3 = np.concatenate((a1[:border], a2[border:])).tolist()
        a4 = np.concatenate((a2[:border], a1[border:])).tolist()

        out.append(a3)
        out.append(a4)

    return np.array(out)


def mutation(population):
    for e in population:
        r = np.random.random()
        if r <= MUTATION_P:
            b = random.randint(0, 11)
            e[b] = not e[b]


population, fenotype, p = represent()
solutions = []
for _ in range(10):
    # select
    new_population = []
    for _ in range(POPSIZE):
        r = np.random.random()
        s = 0
        for i in range(len(p)):
            s += p[i]
            if r < s:
                new_population.append(population[i].tolist())
                break
    # cross over
    population = crossover(new_population)
    # mutation
    mutation(population)

    # update fenotype and p
    for i in range(len(population)):
        fenotype[i] = genotypeToFenotype(population[i])

    fitness = getFitness(fenotype)
    p = fitness / fitness.sum()

    inds = fitness.tolist().index(max(fitness))
    solutions.append((max(fitness), fenotype[inds]))

x = np.arange(-1, 2, 0.01)


graph_y = x * np.sin(10 * np.pi * x) + 2
point_y = np.full_like(x, None)
best_solution = max(solutions)
point_y[int((best_solution[1] + 1) // 0.01)] = best_solution[0]
plt.plot(x, graph_y)
plt.plot(x, point_y, marker='o')
plt.show()

print(f"x = {best_solution[1]}\ny = {best_solution[0]}")