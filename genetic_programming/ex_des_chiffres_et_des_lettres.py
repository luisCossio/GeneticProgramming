import Genetic_program as gp
import Population as pl
import matplotlib.pyplot as plt

def main(args):
    population_size = args.population
    inputs = [int(i) for i in args.inputs]
    output = args.output
    mutation = args.mutation
    epochs = args.epochs

    population = pl.population_des_chiffres(inputs,output,population_size,Mutation = mutation)
    if args.end_condition == 0:
        genetic_algorithm = gp.Genetic_programming(population, condition = gp.end_with_error_zero, iterations = epochs,
                                                iter_method = False)
    else:
        genetic_algorithm = gp.Genetic_programming(population, iterations = epochs, iter_method = True)

    best_fitness_per_epoch, average_fitness_per_epoch, best = genetic_algorithm.run()
    print("best error: ", best.get_fitness())
    print("average result: ", average_fitness_per_epoch[-1])
    # for i in range(len(average_fitness_per_epoch)):
    #     if average_fitness_per_epoch[i]<-100:
    #         average_fitness_per_epoch[i] = -100
    time = [i for i in range(len(best_fitness_per_epoch))]
    plt.plot(time, best_fitness_per_epoch,label=  'best')
    plt.plot(time, average_fitness_per_epoch,label=  'average')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Fitness [Error]')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Genetic algorithm trainnig for WBC detection')
    parser.add_argument('--population', default=25, help='population size',type=int)
    parser.add_argument('--inputs', default=[1], nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--output', default=1, help='desired chiffre', type=int)
    parser.add_argument('--mutation', default=0.1, help='population size', type=float)
    parser.add_argument('--epochs', default=50, help='number of total epochs to run',type=int)
    parser.add_argument('--end-condition', default=0, help='method to end genetic process, 0 for condition method.',type=int)
    # part if the training dataset. default -1 wich means use all dataset.
    args = parser.parse_args()

    main(args)


# class Namespace:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#
#
# args = Namespace(population=50,  # 5, 15 50
#                  inputs = [1,2,3,4,5,6,7,8,9,10,11,12,13,0.5],
#                  output = 177,
#                  mutation=0.1,  # 0.01, 0.1, 0.4
#                  epochs=50,
#                  end_condition=0)
#
# # 0_0_ = population 5, mutation = 0.01
# # 1_0 = population 5, mutation = 0.1
#
# main(args)
