import random
import string
from typing import List, Any

import Individuals as In


class Population_base:
    population: List[Any]

    def __init__(self):
        """
        Base class for population objects. Populations manages all the mechanics of group of individuals such as
        breeding, keep track of fitness stats and define how cross-over and mutation rate are done, given the
        some basic parameters, such as learning rate.
        """
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

    __inputs: List[int]
    output: int
    population: List[In.des_chiffres_root_node]
    mutation_rate: float
    best_des_chiffres: str
    elitism: bool

    def __init__(self, inputs, output, n_population=100, n_tournament=5, Mutation = 0.2, elitism = True):
        """
        Class to manage the population of des_chiffres_root_nodes.

        Args:

            inputs (List(int)): List of inputs to
            output (int): desired output value of the expression. Can be a float.
            n_population (int): number of individuals in population
            n_tournament (int): number of random sampled individuals in the population to choose parent in cross-over
            Mutation (float): mutation rate.
        """

        super().__init__()
        self.n_gen = len(inputs)
        self.__inputs = inputs
        self.n_participants = n_tournament
        self.n_population = n_population
        self.output = output
        self.population = self.generate_random_population(n_population)
        self.best_des_chiffres = None
        self.mutation_rate = Mutation
        self.elitism = elitism

        # self.best_score = 0
        # print(self.population[0])
        # print(self.genes)

    def generate_random_population(self, N_population):
        """
        Method to generate a random population of des_chiffres_root_node
        Args:
            N_population (int): number of individuals

        Returns:
            List(des_chiffres_root_node)
        """
        population = []
        for i in range(N_population):
            population.append(self.generate_random_des_chiffres())
        return population

    def generate_random_des_chiffres(self):
        """
        Method to generate an individual des_chiffres_root_node. Its defined as a index node with a random
        input.

        Returns:
            In.des_chiffres_root_node
        """
        index_random = random.randint(0,len(self.__inputs)-1)
        return In.des_chiffres_root_node(self.__inputs.copy(),self.output ,index_random)

    def new_generation(self):
        """
        Method to produce a new population from the initial one, and replace it.

        Returns:
            None
        """
        new_population = []
        if self.elitism:
            # print("elitism")
            # print("best result thus far: ",self.best_individual.report())
            new_population += [self.best_individual.copy()]
            for i in range(1,self.n_population):
                new_population.append(self.breed_new_individial())
            self.population = new_population
        else:
            for i in range(self.n_population):
                new_population.append(self.breed_new_individial())
            self.population = new_population

    def pick_random_sample(self):
        """
        Method that chooses 'n_participants' in the population randomly.
        Returns: List(In.des_chiffres_root_node)
        """
        sub_sample = []
        for i in range(self.n_participants):
            sub_sample += random.sample(self.population, 1)
        return sub_sample

    def breed_new_individial(self):
        """
        Method that produce a new individual, from the reproduction of 2 parents, selected by the tournament method.
        Cross-over and mutation happen here.

        Returns:
            In.des_chiffres_root_node
        """
        individual1 = self.tournament()
        individual2 = self.tournament()
        new_individual = self.cross_over(individual1, individual2)
        if random.uniform(0,1)<self.mutation_rate:
            return self.mutation(new_individual)
        return new_individual

    def tournament(self):
        """
        Method to select the best individual in a list of individuals, selected randomly.

        Returns:
            In.des_chiffres_root_node
        """
        samples = self.pick_random_sample()
        best_fit = -1
        champion = -1
        for i in range(self.n_participants):
            # samples[i].fitness()
            if samples[i].get_fitness() > best_fit:
                champion = i
                best_fit = samples[i].get_fitness()
        return samples[champion]

    def cross_over(self, individual1, individual2):
        """
        Cross over operation of 2 In.des_chiffres_root_node individuals, also known as parents.
        Returns the offspring of the 2 parents.

        Args:
            individual1 (In.des_chiffres_root_node):
            individual2 (In.des_chiffres_root_node):

        Returns:
            In.des_chiffres_root_node

        """
        return individual1.cross_over(individual2)

    def mutation(self, Individual):
        """
        Method to order a mutation in the individual given.
        Args:
            Individual (In.des_chiffres_root_node): individual that will mutate.
        Returns:
            In.des_chiffres_root_node
        """
        Individual.mutation()
        return Individual

    def calculate_best_score(self):
        """
        Method that calculate the fitness on all individuals in the population, and returns the maximum fitness.

        Returns:
            float:
        """
        score = -10000
        for i in range(self.n_population):
            self.population[i].fitness()
            if score < self.population[i].get_fitness():
                score = self.population[i].get_fitness()
                self.best_des_chiffres = self.population[i].report()
                self.set_best_individual(self.population[i])
        self.best_score = score
        return self.best_score


    def show_answer(self):
        """
        Method to print best solution thus far.
        Returns:
            None
        """
        print("Best score: ",self.best_score)
        print("Winner expresion: ", self.best_des_chiffres)



class population_function_estimation(Population_base):
    __inputs: List[List[float]]
    best_function: In.function_estimation_root_node
    output: int
    population: List[In.function_estimation_root_node]

    def __init__(self, inputs, output, n_population=100, n_toornament=5, Mutation = 0.2, elitism = True):
        """
        Class to manage the population of function_estimation_nodes. Recives an input and output of the form:
        inputs = [x1,x2,x3,x4,..,xn]
        output = [y1,y2,y3,y4,..,yn]

        Args:
            inputs (List(int)): List of inputs to
            output (int): desired output value of the expression. Can be a float.
            n_population (int): number of individuals in population
            n_tournament (int): number of random sampled individuals in the population to choose parent in cross-over
            Mutation (float): mutation rate.
            elitism (bool): True if use Elitism during each process of creating the next generation of individuals.
        """

        super().__init__()
        self.n_gen = len(inputs)
        self.__inputs = self.get_inputs_representation(inputs)
        self.n_tournament_participants = n_toornament
        self.n_population = n_population
        self.output = output
        self.population = self.generate_random_population(n_population)
        self.best_function = None
        self.mutation_rate = Mutation
        self.elitism = elitism


    def generate_random_population(self, N_population):
        """
        Method to generate a random population of function_estimation_root_node's
        Args:
            N_population (int): number of individuals

        Returns:
            List(In.function_estimation_root_node)
        """
        population = []
        for i in range(N_population):

            population.append(self.generate_random_function_estimation_node())
        return population

    def generate_random_function_estimation_node(self):
        """
        Method to generate an individual function_estimation_root_node. Its defined as a index node with a random
        input.

        Returns:
            In.function_estimation_root_node
        """
        index_random = random.randint(0,len(self.__inputs)-1)
        return In.function_estimation_root_node(self.__inputs.copy(), self.output.copy(), index_random)

    def new_generation(self):
        """
        Method to produce a new population from the initial one, and replace it.

        Returns:
            None
        """
        new_population = []
        if self.elitism:
            # print("elitism")
            # print("best result thus far: ",self.best_individual.report())
            new_population += [self.best_individual.copy()]
            for i in range(1,self.n_population):
                new_population.append(self.breed_new_individial())
            self.population = new_population
        else:
            for i in range(self.n_population):
                new_population.append(self.breed_new_individial())
            self.population = new_population

    def pick_random_sample(self):
        """
        Method that chooses 'n_participants' in the population randomly and returns them
        Returns:
            List(In.function_estimation_root_node)
        """
        sub_sample = []
        for i in range(self.n_tournament_participants):
            sub_sample += random.sample(self.population, 1)
        return sub_sample

    def breed_new_individial(self):
        """
        Method that produce a new individual, from the reproduction of 2 parents, selected by the tournament method.
        Cross-over and mutation happen here.

        Returns:
            In.function_estimation_root_node
        """
        individual1 = self.tournament()
        individual2 = self.tournament()
        new_individual = self.cross_over(individual1, individual2)
        if random.uniform(0,1)<self.mutation_rate:
            return self.mutation(new_individual)
        return new_individual

    def tournament(self):
        """
        Method to select the best individual in a list of individuals, selected randomly.

        Returns:
            In.function_estimation_root_node
        """
        samples = self.pick_random_sample()
        best_fit = -1
        champion = -1
        for i in range(self.n_tournament_participants):
            # samples[i].fitness()
            if samples[i].get_fitness() > best_fit:
                champion = i
                best_fit = samples[i].get_fitness()
        return samples[champion]

    def cross_over(self, individual1, individual2):
        """
        Cross over operation of 2 In.function_estimation_root_node individuals, also known as parents.
        Returns the offspring of the 2 parents.

        Args:
            individual1 (In.function_estimation_root_node):
            individual2 (In.function_estimation_root_node):

        Returns:
            In.function_estimation_root_node
        """
        return individual1.cross_over(individual2)

    def mutation(self, Individual):
        """
        Method to order a mutation in the individual given.
        Args:
            Individual (In.function_estimation_root_node): individual that will mutate.
        Returns:
            In.function_estimation_root_node
        """
        Individual.mutation()
        return Individual

    def calculate_best_score(self):
        """
        Method that calculate the fitness on all individuals in the population

        Returns: max fitness in the population
            float:
        """
        score = -10000
        for i in range(self.n_population):
            self.population[i].fitness()
            if score < self.population[i].get_fitness():
                score = self.population[i].get_fitness()
                self.best_function = self.population[i].report()
                self.set_best_individual(self.population[i])
        self.best_score = score
        return self.best_score


    def show_answer(self):
        """
        Method to print best result

        Returns:
            None
        """
        print("Best score: ",self.best_score)
        print("Winner expresion: ", self.best_function)

    def get_inputs_representation(self, inputs):
        """
        Method to convert a series of inputs into a representation valid for function estimation.
        input = [x1,x2,x3,x4]
        returns input of the form:
        new_input = [[c1,c1,c1,c1], [c2,c2,c2,c2],....,[cm,cm,cm,cm],[x1,x2,x3,x4],[x1,x2,x3,x4],[x1,x2,x3,x4]
        [x1,x2,x3,x4],..,[x1,x2,x3,x4]]

        Args:
            inputs(List(int)):

        Returns:
            List(List(float)):
        """
        basic_inputs = [0.01,0.1,1,2,3,4,5,6,7,8,9,10,100]
        size_inputs = len(inputs)
        number_iteration_initial_input = 10
        new_inputs = [[basic_inputs[i]]*size_inputs for i in range(len(basic_inputs))]
        new_inputs += [inputs.copy() for i in range(number_iteration_initial_input)]
        return new_inputs

