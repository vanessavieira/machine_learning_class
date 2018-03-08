# Aprendizagem de Máquina

# Algorítmo Genético

# Geração da População - tamanho: 120; array 0->5, cada posição 0->360 (0->90, 0->180, 0->270, 0-> 360)
# Gene - 6 angulos
# Função de Adaptação (fitness) - request servidor (url)
# Seleção - rankeia a população, 2/3 do topo sobrevivem, escolhe x indivíduos do 1/3 de baixo para sobreviver
# Cruzamento - 33 filhos, cruza random1 com random2, filho(i) = (random1(i) + random2(i))/2
# Mutação - escolhe id 0->5, antena(id) += random de 0->360

# Choosing a fitness function

import random
import operator
import requests
import sys

def fitness(antenna):

    response = requests.get("http://localhost:8080/antenna/simulate?" +
                            "phi1=" + str(antenna[1]) +
                            "&theta1=" + str(antenna[2]) +
                            "&phi2=" + str(antenna[3]) +
                            "&theta2=" + str(antenna[4]) +
                            "&phi3=" + str(antenna[5]) +
                            "&theta3=" + str(antenna[6]))
    #print(response.content)

    return response.content

# Creating our individuals


def generateAntenna ():
	i = 1
	currentAntenna = []
	while i <= 7:
		angle = int(360 * random.random())
		currentAntenna.append(angle)
		i +=1
	return currentAntenna

def generateFirstPopulation(sizePopulation):
	population = []
	i = 0
	while i < sizePopulation:
		population.append(generateAntenna())
		i+=1
	return population

# From one generation to the next

# # Breeders selection

def computePerfPopulation(population, password):
	populationPerf = {}
	for individual in population:
		populationPerf[individual] = fitness(password, individual)
    # Ranking population
	return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)

def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])
	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])
	random.shuffle(nextGeneration)
	return nextGeneration

# # Breeding

def createChild(individual1, individual2):
	child = ""
	for i in range(len(individual1)):
		if (int(100 * random.random()) < 50):
			child += individual1[i]
		else:
			child += individual2[i]
	return child

def createChildren(breeders, number_of_child):
	nextPopulation = []
	for i in range(len(breeders)/2):
		for j in range(number_of_child):
			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
	return nextPopulation

# # Mutation

def mutateWord(word):
    index_modification = int(random.random() * len(word))
    if (index_modification == 0):
        word = chr(97 + int(26 * random.random())) + word[1:]
    else:
        word = word[:index_modification] + chr(97 + int(26 * random.random())) + word[index_modification + 1:]
    return word


def mutatePopulation(population, chance_of_mutation):
    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutateWord(population[i])
    return population

def main(argv):
    gains = []
    population = generateFirstPopulation(120)
    i = 0
    while(i < len(population)):
        s = fitness(population[i])
        population[i][0] = (str(s).split("\\n")[0][2:])
        print(population[i])
        i += 1


if __name__ == "__main__":
    main(sys.argv)