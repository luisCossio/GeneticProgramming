import random
import string
from typing import List, Any

import numpy as np
import Individuals as In


class Population_base:
    population: List[In.individual]

    def __init__(self):
        self.best_score = 0
        self.population = []
        self.mutation_rate = 0.1
        self.best_individual = None

    def get_best_score(self):
        return self.best_score

    def show_answer(self):
        pass

    def calculate_best_score(self):
        pass

    def get_average_fitness(self):
        average = 0
        for i in range(len(self.population)):
            average += self.population[i].get_fitness()
        return average/len(self.population)

    def get_best_individuals(self):
        return self.population[:6]

    def get_best_individual(self):
        return self.best_individual

    def set_best_individual(self,Ind):
        self.best_individual = Ind


class population_des_chiffres(Population_base):
    population: List[In.des_chiffres_node]

    def __init__(self, inputs, n_population=100, n_random_cross_over=5, Mutation = 0.2):
        """
        Class to manage the population of datasets.

        Args:
            inputs (List):
            n_population:
            n_random_cross_over:
            Mutation:
        """

        super().__init__()
        self.n_gen = len(inputs)
        self.__inputs = inputs
        self.n_random_tournament = n_random_cross_over
        self.n_population = n_population
        self.population = self.generate_random_population(n_population)
        self.best_des_chiffres = None
        self.mutation_rate = Mutation
        # self.best_score = 0
        # print(self.population[0])
        # print(self.genes)

    def generate_random_population(self, N_population):
        population = []
        for i in range(N_population):
            population.append(self.generate_random_des_chiffres(self.n_gen))
        return population

    def generate_random_des_chiffres(self, n_gen):
        return In.des_chiffres_node(self.__inputs.copy())

    def new_generation(self):
        new_population = []
        for i in range(self.n_population):
            new_population.append(self.breed_new_individial())
        self.population = new_population

    def pick_random_sample(self):
        sub_sample = []
        for i in range(self.n_random_tournament):
            sub_sample += random.sample(self.population, 1)
        return sub_sample

    def breed_new_individial(self):
        individual1 = self.tournament()
        individual2 = self.tournament()
        new_individual = self.cross_over(individual1, individual2)
        if np.random.random()<self.mutation_rate:
            return self.mutation(new_individual)
        return new_individual

    def tournament(self):
        samples = self.pick_random_sample()
        best_fit = -1
        champion = -1
        for i in range(self.n_random_tournament):
            samples[i].fitness()
            if samples[i].get_fitness() > best_fit:
                champion = i
                best_fit = samples[i].get_fitness()
        return samples[champion]

    def cross_over(self, individual1, individual2):
        return individual1.cross_over(individual2)

    def mutation(self, Individual):
        # random_int = random.randint(0, self.n_gen)
        # random_letter = self.genes[random_int1]
        Individual.mutation()
        return Individual

    def calculate_best_score(self):
        score = 0
        for i in range(self.n_population):
            self.population[i].fitness()
            if score < self.population[i].get_fitness():
                score = self.population[i].get_fitness()
                self.best_des_chiffres = self.population[i].get_result()
                self.set_best_individual(self.population[i])
        self.best_score = score
        return self.best_score


    def show_answer(self):
        print("Best score: ",self.best_score)
        print("Winner expresion: ", self.best_des_chiffres)