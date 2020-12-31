import random
import numpy as np
from typing import List, Any


class des_chiffres_node:

    # node1: des_chiffres_node
    # node2: des_chiffres_node

    def __init__(self, input_index,input_value, type_node = 1, Node1 = None, Node2 = None):
        self.type_node = type_node  # 0 for null node, 1 for input node, 2 for operation node

        self.mutation_swap_method = None
        self.mutation_method = None
        self.result_calculation = None
        self.operation = None

        if type_node == 0:
            self.input_index = None
            self.input = None
            self.value = 0
            self.node1 = None
            self.node2 = None

        elif type_node == 1:
            self.input_index = input_index
            self.input = input_value
            self.value = 1
            self.mutation_method = self.mutate_input_node
            self.result_calculation = self.get_result_input_node
            self.mutation_swap_method = self.mutate_swap_input_node
            self.node1 = None
            self.node2 = None

        else:
            assert ( isinstance(Node2,des_chiffres_node) and isinstance(Node1,des_chiffres_node)),\
                ("Error. Invalid type of node")
            self.input_index = None
            self.input = None
            self.value = 3
            self.mutation_method = self.mutate_operation_node
            self.mutation_swap_method = self.mutate_swap_operation_node
            self.operation = self.get_random_operation()
            self.result_calculation = self.get_result_operation_node
            self.node1 = Node1
            self.node2 = Node2

    def get_value(self):
        return self.value

    def calculate_result(self):
        return self.result_calculation()

    def mutate_remaining_inputs(self, node_index, input_index, input_value):
        """
        Method of mutation when there is still some inputs not yet included on the three

        Args:
            node_index (int): number of node where for a give tree
            input_index (int): index employed in cased is used

        Returns: (int) number of nodes added. 0, 1 or 2
        """
        return self.mutation_method(node_index,input_index, input_value)



    def mutate_input_node(self, node_index, input_index, input_value):
        if node_index == self.value:
            self.type_node = 2
            self.node1 = des_chiffres_node(self.input_index,self.input)
            self.node2 = des_chiffres_node(input_index,input_value)
            self.value = 3
            self.input_index = None
            self.input = None
            self.operation = self.get_random_operation()
            self.result_calculation = self.get_result_operation_node
            self.mutation_method = self.mutate_operation_node
            self.mutation_swap_method = self.mutate_swap_operation_node
            return 2
        else:
            print("Error: invalid return reached. Node index: ",node_index)  # Delete if
            return 0

    def mutate_operation_node(self, node_index, input_index, input_value):
        if self.node1.get_value() >= node_index:
            result = self.node1.mutate_remaining_inputs(node_index, input_index, input_value)
            # if result > 0:
            self.value += result
            return result

        elif self.node1.get_value()+1 == node_index:
            self.operation = self.get_random_operation()
            return 0

        else:
            result = self.node2.mutate_remaining_inputs(node_index-(self.node1.get_value()+1), input_index, input_value)
            # if result > 0:
            self.value += result
            return result

    def cross_over(self, new_individual):
            base = self.copy()
            return base

    def copy(self):
        if self.type_node == 1:
            return des_chiffres_node(self.input_index,self.input)
        elif self.type_node == 0:
            return des_chiffres_node(None, None, type_node=0)
        elif self.type_node == 2:
            node_son1 = self.node1.copy()
            node_son2 = self.node2.copy()
            new_node = des_chiffres_node(self.input_index,self.input, type_node=2,Node1=node_son1,Node2=node_son2)
            return new_node
        else:
           print("Error. Invalid type of node", self.type_node)


    def get_random_operation(self):
        random_value = random.uniform(0, 1)
        if random_value<0.25:
            return sum_op()
        elif random_value<0.5:
            return rest_op()
        elif random_value<0.75:
            return mult_op()
        elif random_value<=1:  # Delete this
            return div_op()
        else:
            print("Error, invalid random number in operation generator: ",random_value)

    def get_result_operation_node(self):
        if self.node1.type_node == 0:
            return self.operation(self.operation.get_null_value(),self.node2.calculate_result())
        elif self.node2.type_node == 0:
            return self.operation(self.node1.calculate_result(),self.operation.get_null_value())
        return self.operation(self.node1.calculate_result(),self.node2.calculate_result())

    def get_result_input_node(self):
        return self.input

    # def mutate_by_swapping_inputs(self, index_to_find, indexes_swap, indexes_swap):
    #     """
    #     Method of mutation if there are no longer inputs to assign to this tree
    #
    #     Returns: (bool) True if the node was found
    #     """
    #     return self.mutation_swap_method(index_to_find, indexes_swap, indexes_swap)

    def mutate_swap_input_node(self, indexes_swap, inputs_value_swap):
        for i,input_index in enumerate(indexes_swap):
            if input_index == self.input_index:
                self.input_index = indexes_swap[(i+1)%2]
                self.input = inputs_value_swap[(i+1)%2]
                return 1
        return 0

    def mutate_swap_operation_node(self, indexes_swap, inputs_swap):
        if self.node1.type_node == 0:

            nodes_found = self.node2.mutation_swap_method(indexes_swap, inputs_swap)
            return nodes_found

        found_node = self.node1.mutation_swap_method( indexes_swap, inputs_swap)

        if found_node==2:
            return 2

        if self.node2.type_node == 0:
            return found_node

        found_node += self.node2.mutation_swap_method(indexes_swap, inputs_swap)
        return found_node


class des_chiffres_root_node(des_chiffres_node):


    __fitness: int
    remaining_inputs: List[int]

    def __init__(self,inputs, desired_value, input_index=0,type_node = 1,**kwargs):
        """

        Args:
            input_index (int):
            inputs (List[ints]):
        """
        super().__init__(input_index=input_index,input_value=inputs[input_index], type_node=type_node, **kwargs)
        self.remaining_inputs = inputs
        self.back_up_inputs = inputs.copy()
        self.indexes_input = [i for i in range(len(inputs))]
        self.remaining_inputs.pop(input_index)
        self.indexes_input.pop(input_index)

        self.__fitness = -10000
        self.__number_inputs = len(inputs)
        self.desired_output = desired_value

    def copy(self):
        if self.type_node == 1:
            return des_chiffres_root_node(self.back_up_inputs.copy(),self.desired_output,self.input_index)
        # elif self.type_node == 0:
        #     return des_chiffres_root_node(None, None, type_node=0)
        elif self.type_node == 2:
            node_son1 = self.node1.copy()
            node_son2 = self.node2.copy()
            new_node = des_chiffres_root_node(self.back_up_inputs.copy(),self.desired_output,self.input_index,
                                              type_node=2,Node1=node_son1,Node2=node_son2)
            return new_node
        else:
           print("Error. Invalid type of node", self.type_node)

    def mutation(self):
        number_nodes = self.get_value()
        if len(self.remaining_inputs) > 0:
            random_node = random.randint(1,number_nodes)
            random_index = random.randint(0,len(self.remaining_inputs)-1)
            result = self.mutate_remaining_inputs(random_node,self.indexes_input[random_index],self.remaining_inputs[random_index])
            # self.sum_value()
            if result > 0:
                self.remaining_inputs.pop(random_index)
                self.indexes_input.pop(random_index)

        else:
            random_index1 = random.randint(0, self.__number_inputs)
            random_index2 = random.randint(0, self.__number_inputs)

            _ = self.mutation_swap_method([random_index1, random_index2],
                                               [self.back_up_inputs[random_index1],
                                                self.back_up_inputs[random_index2]])
            if _ != 2:
                print("Error. Impossible condition achieved")

            # self.sum_value()
        # pass

    def fitness(self):
        result = self.get_result_operation_node()



class operation:
    def __init__(self):
        pass

    def get_null_value(self):
        return 0

class sum_op(operation):
    def __call__(self, *args):
        return args[0] + args[1]

class rest_op(operation):
    def __call__(self, *args):
        return args[0]-args[1]

class mult_op(operation):
    def __call__(self, *args):
        return args[0]*args[1]

    def get_null_value(self):
        return 1

class div_op(operation):
    def __call__(self, *args):
        return args[0]/args[1]

    def get_null_value(self):
        return 1
