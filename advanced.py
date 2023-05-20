import random

import numpy as np
import matplotlib.pyplot as plt

LOW = -1
HIGH = 2
MUTATION_P = 0.5
POPSIZE = 30


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

        points = [a1, a2, a3, a4]
        fenotype = np.empty(4, np.float64)
        for i in range(4):
            fenotype[i] = genotypeToFenotype(points[i])
        fitness = getFitness(fenotype).tolist()

        i = fitness.index(max(fitness))
        out.append(points[i])
        fitness.remove(fitness[i])
        points.remove(points[i])
        i = fitness.index(max(fitness))
        out.append(points[i])

    return np.array(out)


def mutation(population):
    for e in population:
        r = np.random.random()
        if r <= MUTATION_P:
            b = random.randint(0, 11)
            e[b] = not e[b]


population, fenotype, p = represent()
solutions = []  # it will contain (fitness,fenotype)
search_num = 5
z = 0
while search_num > 0:
    # select
    z += 1
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
    if len(solutions) > 0:
        if max(fitness) > solutions[-1][0]:
            inds = fitness.tolist().index(max(fitness))
            solutions.append((max(fitness), fenotype[inds]))
            search_num = 5
        elif max(fitness) < solutions[-1][0]:
            search_num -= 1
    else:
        inds = fitness.tolist().index(max(fitness))
        solutions.append((max(fitness), fenotype[inds]))

x = np.arange(-1, 2, 0.01)
graph_y = x * np.sin(10 * np.pi * x) + 2
point_y = np.full_like(x, None)
point_y[int((solutions[-1][1] + 1) // 0.01)] = solutions[-1][0]
plt.plot(x, graph_y)
plt.plot(x, point_y, marker='o')

plt.show()

print(f"maximum value finded is :\n x = {solutions[-1][1]}\n y = {solutions[-1][0]}")
print(f"number of main loop run : {z}")
