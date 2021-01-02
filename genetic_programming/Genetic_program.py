# import string
# import random
# from typing import List


import Population as pl



class Genetic_programming:
    population: pl.Population_base

    def __init__(self, Population, condition=None, iterations=100, iter_method=False):

        """

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

        :return:
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

    Args:
        population (pl.population_des_chiffres):
    """
    best_fitness = population.best_individual.get_fitness()
    return not best_fitness == 0


def main(args):
    population_size = args.population
    inputs = args.inputs
    output = args.output
    mutation = args.mutation
    epochs = args.epochs
    args.end_condition
    population = pl.population_des_chiffres(inputs,output,population_size,Mutation = mutation)
    if args.end_condition == 0:
        genetic_algorithm = Genetic_programming(population, condition = end_with_error_zero, iterations = epochs,
                                                iter_method = False)
    else:
        genetic_algorithm = Genetic_programming(population, iterations = epochs, iter_method = True)

    best_fitness_per_epoch, average_fitness_per_epoch, best = genetic_algorithm.run()
    print("best: ", best.get_fitness())
    print("averge result: ", average_fitness_per_epoch[-1])


#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Genetic algorithm trainnig for WBC detection')
#     parser.add_argument('--population', default=25, help='population size',type=int)
#     parser.add_argument('--inputs', default=[1], help='inputs values', type=list)
#     parser.add_argument('--output', default=1, help='desired chiffre', type=int)
#     parser.add_argument('--mutation', default=0.1, help='population size', type=float)
#     parser.add_argument('--epochs', default=50, help='number of total epochs to run',type=int)
#     parser.add_argument('--end-condition', default=0, help='method to end genetic process, 0 for condition method.',type=int)
#     # part if the training dataset. default -1 wich means use all dataset.
#     args = parser.parse_args()
#
#     main(args)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args = Namespace(population=50,  # 5, 15 50
                 inputs = [1,2,3,4,5,6,7,8],
                 output = 23,
                 mutation=0.1,  # 0.01, 0.1, 0.4
                 epochs=50,
                 end_condition=0)

# 0_0_ = population 5, mutation = 0.01
# 1_0 = population 5, mutation = 0.1

main(args)
