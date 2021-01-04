# import string
# import random
# from typing import List


import Population as pl



class Genetic_programming:
    population: pl.Population_base

    def __init__(self, Population, condition=None, iterations=100, iter_method=False):

        """
        Class to implement genetic programming

        :type Population: Populations.Population_base
        :param Population: [Population]
        :param condition: [function]
        :param iterations: [int]
        :param iter_method: [bool]
        """
        self.end_condition = condition
        self.population = Population
        self.evolution_score = []
        self.iterations = iterations
        self.end_by_time = iter_method
        if condition is None:
            assert iter_method, "Invalid combination of iteration method and end by a number of epochs given. Only one can be done"

    def run(self):
        """
        Method to iterate make a population run through an evolution process until a certain condition is meet. If
        end_by_time is True, then the program ends when a certain number of iterations has been done. If not, then
        a condition is used to iterate, that returns True until such condition is meet, for example error 0, or until
        all iterations are done.
        :return (None):
        """
        best_scores = []
        average_scores = []
        # self.population.new_generation()
        best_scores += [self.population.calculate_best_score()]
        average_scores += [self.population.get_average_fitness()]
        print("initial score: ", best_scores[-1])
        if self.end_by_time:  # condition to end in a given number of iterations
            for i in range(self.iterations):
                self.population.new_generation()
                best_scores += [self.population.calculate_best_score()]
                average_scores += [self.population.get_average_fitness()]


        else:
            counter = 0
            while self.end_condition(self.population) and counter < self.iterations:
                # condition to end when a condition is reached.
                self.population.new_generation()
                best_scores += [self.population.calculate_best_score()]
                average_scores += [self.population.get_average_fitness()]
                counter += 1
            print("iteration: ", counter)
        self.population.show_answer()
        # best_samples = self.population.get_best_individuals()
        # average = self.population.get_average_fitness()
        best = self.population.get_best_individual()
        return best_scores, average_scores, best

def end_with_error_zero(population):
    """
    Condition to end a run process when error 0 is achieved.
    Args:
        population (pl.population_base):
    """
    best_fitness = population.best_individual.get_fitness()
    return not best_fitness == 0




