import random
from typing import List, Any


class des_chiffres_node:

    # operation: operation
    type_node: int
    # node1: des_chiffres_node
    # node2: des_chiffres_node

    def __init__(self, input_index, input_value, type_node = 1, operation_node = None, Node1 = None, Node2 = None):
        """
        Class that defines a node in a tree, that defines a solution for a problem of des chiffres et des lettres.
        Each node can be either input node, operation node or null node. Input nodes contains an input value.
        Operation nodes contain an operation (+, -, * or /) and calculate the result of such operation between
        the results given by its two nodes sons. Null node simple exist to mantain a binary structure and when a
        node es nullify.

        Args:
            input_index (int): index of the input in the list of all inputs. Used as an index to distinguished inputs
            input_value (float): input value
            type_node (int): type of node
            operation_node (operation): operation object
            Node1 (des_chiffres_node): node son 1
            Node2 (des_chiffres_node): node son 2
        """
        self.type_node = type_node  # 0 for null node, 1 for input node, 2 for operation node
        self.mutation_swap_method = None  # function
        self.mutation_method = None  # function
        self.result_calculation = None  # function
        self.operation = None
        self.get_sub_tree = None  # function

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
            self.get_sub_tree = self.get_sub_tree_input_node

        else:

            assert (isinstance(Node2,des_chiffres_node) and isinstance(Node1,des_chiffres_node)),\
                ("Error. Invalid type of node {}".format(type(Node2)))
            self.input_index = None
            self.input = None

            self.mutation_method = self.mutate_operation_node
            self.mutation_swap_method = self.mutate_swap_operation_node
            self.operation = operation_node
            self.result_calculation = self.get_result_operation_node
            self.node1 = Node1
            self.node2 = Node2
            self.get_sub_tree = self.get_sub_tree_operation_node
            self.value = 0
            self.update_value()


    def get_value(self):
        """
        Method to get the number of inputs in this sub tree.
        Returns: number of nodes in this sub tree. 1 if this is an input node
            int
        """
        return self.value

    def calculate_result(self):
        """
        Method to calculate the result of a given node. If its a input node result_calculation returns the input value,
        if its an operation node, calculate the result of the operation given the 2 node sons. null node can't affect
        the result.

        Returns:

        """
        return self.result_calculation()

    def mutate_remaining_inputs(self, node_index, input_index, input_value):
        """
        Method of mutation when there is still some indexes not yet included on the three

        Args:
            node_index (int): number of node where for a give tree
            input_index (int): index employed in cased is used

        Returns: (int) number of nodes added. 0, 1 or 2
        """
        return self.mutation_method(node_index,input_index, input_value)



    def mutate_input_node(self, node_index, input_index, input_value):
        """
        Method to mutate a node, into an operation node. This node can only be an input node.
        Args:
            node_index (int): index of node in the tree (not used in the part, because this is a end condition
            in a recursive algorithm)
            input_index (int):
            input_value (float):

        Returns: number of nodes added to the tree
            int
        """
        self.type_node = 2
        self.node1 = des_chiffres_node(self.input_index,self.input)
        self.node2 = des_chiffres_node(input_index,input_value)
        self.value = 3
        self.input_index = None
        self.input = None
        self.operation = self.get_random_operation()
        self.result_calculation = self.get_result_operation_node
        self.get_sub_tree = self.get_sub_tree_operation_node
        self.mutation_method = self.mutate_operation_node
        self.mutation_swap_method = self.mutate_swap_operation_node
        return 2


    def mutate_operation_node(self, node_index, input_index, input_value):
        """
        Method to manage a mutation command. The variable node_index is the identifier in the node to be mutated.
         If this node is the one to be mutated the operation field is replaced by another random operation. Else, the
         mutation command is recursively passed down to one of the node sons.
        Args:
            node_index (int): index of node in this tree
            input_index (int):
            input_value (float):
        Returns: number of nodes added to the tree.
            int
        """
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

    # def cross_over(self, new_individual):
    #     base = self.copy()
    #     return base

    def copy(self):
        """
        Method to copy this node.

        Returns:
            des_chiffres_node
        """
        if self.type_node == 1:
            return des_chiffres_node(self.input_index,self.input)
        elif self.type_node == 0:
            return des_chiffres_node(None, None, type_node=0)
        else:
            node_son1 = self.node1.copy()
            node_son2 = self.node2.copy()
            new_node = des_chiffres_node(self.input_index, self.input, type_node=2, operation_node= self.operation.copy(),
                                         Node1=node_son1, Node2=node_son2)
            return new_node


    def get_random_operation(self):
        """
        Method to get a random operation
        Returns:
            operation
        """
        random_value = random.uniform(0, 1)
        if random_value<0.25:
            return sum_op()
        elif random_value<0.5:
            return rest_op()
        elif random_value<0.75:
            return mult_op()
        else:
            return div_op()


    def get_result_operation_node(self):
        """
        Method to calculate the result of this node. Used only for operation nodes.
        Returns:
            float
        """
        if self.type_node==1:
            return self.input
        elif self.node1.type_node == 0:
            return self.operation(self.operation.get_null_value(),self.node2.calculate_result())
        elif self.node2.type_node == 0:
            return self.operation(self.node1.calculate_result(),self.operation.get_null_value())
        return self.operation(self.node1.calculate_result(),self.node2.calculate_result())

    def get_result_input_node(self):
        """
        Method to calculate the result of this node. Used only for input nodes.
        Returns:
            float
        """

        return self.input

    # def mutate_by_swapping_inputs(self, index_to_find, indexes_swap, indexes_swap):
    #     """
    #     Method of mutation if there are no longer indexes to assign to this tree
    #
    #     Returns: (bool) True if the node was found
    #     """
    #     return self.mutation_swap_method(index_to_find, indexes_swap, indexes_swap)

    def mutate_swap_input_node(self, indexes_swap, inputs_value_swap):
        """
        Method to mutate this input node by swapping a given index with another if this input index exist in the
        indexes_swap.

        Args:
            indexes_swap (List(int)):
            inputs_value_swap (List(float)):

        Returns: number of inputs swapped.
            int
        """
        for i,input_index in enumerate(indexes_swap):
            if input_index == self.input_index:
                self.input_index = indexes_swap[(i+1)%2]
                self.input = inputs_value_swap[(i+1)%2]
                return 1
        return 0

    def mutate_swap_operation_node(self, indexes_swap, inputs_swap):
        """
        Method to mutate and input node by swapping a given index with another if this input index exist in the
        indexes_swap. Looks for the node to be swapped and end when 2 inputs have been swapped.

        Args:
            indexes_swap (List(int)):
            inputs_value_swap (List(float)):

        Returns: number of inputs swapped.
            int
        """
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

    def get_sub_tree_operation_node(self,node_index):
        """
        Method to extract a copy of a given sub tree inside this tree. The tree to be returned is of index 'node_index'
        This method is used only for operation nodes.
        Args:
            node_index (int):

        Returns (des_chiffres_node):

        """
        # if self.value>3:
        #     print("detected")
        if self.node1.get_value() >= node_index:
            tree = self.node1.get_sub_tree(node_index)
            return tree

        elif self.node1.get_value() + 1 == node_index:
            tree = self.copy()
            return tree
            # return self.copy()
        else:
            tree = self.node2.get_sub_tree(node_index - (self.node1.get_value() + 1))
            return tree

    def get_sub_tree_input_node(self, node_index):
        """
        Method to extract a copy of this tree/node.
        This method is used only for input nodes.
        Args:
            node_index (int):

        Returns (des_chiffres_node):

        """
        return self.copy()



    def get_indexes_in_tree(self, indexes):
        """
        Method to retrieve the indexes in this tree
        Args:
            indexes (List(int)):

        Returns:

        """
        if self.type_node == 0:
            return indexes

        elif self.type_node == 2:
            new_indexes = self.node1.get_indexes_in_tree(indexes)
            new_indexes = self.node2.get_indexes_in_tree(new_indexes)
            return new_indexes

        else:
            indexes += [self.input_index]
            return indexes

    def nullify_inputs(self, inputs_to_nullify):
        """
        Given a list of inputs indexes, removes any of this in this node or branches of this node.
        Args:
            inputs_to_nullify (List(int)):

        Returns:

        """
        if self.type_node == 0:
            return inputs_to_nullify

        elif self.type_node == 1:
            if self.input_index in inputs_to_nullify:
                for i in range(len(inputs_to_nullify)):
                    if inputs_to_nullify[i] == self.input_index:
                        inputs_to_nullify.pop(i)
                        break
                self.nullify_this_node()
            return inputs_to_nullify

        else:
            inputs_to_nullify = self.node1.nullify_inputs(inputs_to_nullify)
            self.update_value()
            if len(inputs_to_nullify)==0:
                return []
            inputs_to_nullify = self.node2.nullify_inputs(inputs_to_nullify)
            self.update_value()
            return inputs_to_nullify

    def nullify_this_node(self):
        """
        Method to turn this node into a null node

        Returns:
            None
        """
        self.mutation_swap_method = None
        self.mutation_method = None
        self.result_calculation = None
        self.get_sub_tree = None
        self.operation = None
        self.type_node = 0
        self.input_index = None
        self.input = None
        self.value = 0
        self.node1 = None
        self.node2 = None

    def is_node(self, node_number):
        """
        Method to check if this node is the node 'node_number'.
        Args:
            node_number (int): number of the node in the tree

        Returns:
            bool
        """
        if self.type_node == 1:
            if 1 == node_number:
                return True
            return False
        elif self.type_node == 2:
            if self.node1.get_value() >= node_number:
                return False
            elif self.node1.get_value() + 1 == node_number:
                return True
            return False
        elif self.type_node == 0:
            return False
        else:
            print("error invalid condition achieved. Null node ask if is node")

    def replace_branch(self, node_to_be_replace, sub_tree, indexes_to_remove):
        """
        Method to look for a given node to be replace.

        Args:
            node_to_be_replace (int):
            sub_tree (des_chiffres_node):
            indexes_to_remove (List(int)): List of indexes to remove. this value is recursively reduced every time a
            node is found with an index inside this list.

        Returns:
            None
        """

        if self.type_node==0 or self.type_node==1:
            return

        elif self.node1.is_node(node_to_be_replace):
            self.node1 = sub_tree
            self.update_value()
            _ = self.node2.nullify_inputs(indexes_to_remove)
            return

        elif self.node2.is_node(node_to_be_replace-(self.node1.get_value() + 1)):
            self.node2 = sub_tree
            self.update_value()
            _ = self.node1.nullify_inputs(indexes_to_remove)
            return

        elif self.node1.get_value() >= node_to_be_replace:
            self.node1.replace_branch(node_to_be_replace, sub_tree, indexes_to_remove)
            _ = self.node2.nullify_inputs(indexes_to_remove)

        else:
            self.node2.replace_branch(node_to_be_replace - (self.node1.get_value() + 1), sub_tree, indexes_to_remove)
            _ = self.node1.nullify_inputs(indexes_to_remove)
        self.update_value()

    def fix_null_nodes(self):
        """
        Method used after cross-over operation to fix certain invalid configurations
        Returns:

        """
        if self.type_node==2:
            self.node1.fix_null_nodes()
            self.node2.fix_null_nodes()
            if self.node1.type_node == 0 and self.node2.type_node == 0:
                self.nullify_this_node()
        if self.type_node == 2:
            self.value = self.node1.get_value()+self.node2.get_value()+1

    def report(self):
        """
        Method to get recursively the expression of this tree.
        Returns:

        """
        if self.type_node == 1:
            return str(self.input)

        elif self.type_node == 2:
            report1 = self.node1.report()
            report2 = self.node2.report()
            if report1 == "":
                if report2 == "":
                    return ""
                else:
                    operation_result = self.operation.report()
                    if operation_result == '-':
                        return "({:s}{:s}) ".format(operation_result,
                                                          self.node2.report())
                    elif operation_result == '/':
                        return "({:d} {:s} {:s}) ".format(self.operation.get_null_value(), operation_result,
                                                          self.node2.report())
                    return report2
            else:
                if report2 == "":
                    return report1
                else:
                    return "({:s} {:s} {:s}) ".format(self.node1.report(),self.operation.report(),self.node2.report())
        else:
            return ""

    def update_value(self):
        """
        Update value of node
        Returns:

        """
        self.value = self.node2.get_value()+1+self.node1.get_value()


class function_estimation_node(des_chiffres_node):
    number_inputs: int

    def __init__(self, input_index, input_value, type_node=1, n_inputs = 0, operation_node=None, Node1=None, Node2=None):
        """
        Node for the task of function estimation. Employs the same mechanics of des_chiffres_node.
        Args:
            input_index (int): index of input in the main list of inputs. Works as an identifier of inputs.
            input_value (List(float)): input value
            type_node (int): type of this node
            n_inputs (int): number inputs in total
            operation_node (operation):
            Node1 (function_estimation_node):
            Node2 (function_estimation_node):
        """
        self.number_inputs = n_inputs
        self.type_node = type_node  # 0 for null node, 1 for input node, 2 for operation node

        self.mutation_swap_method = None
        self.mutation_method = None
        self.result_calculation = None
        self.operation = None
        self.get_sub_tree = None

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
            self.get_sub_tree = self.get_sub_tree_input_node

        else:

            assert (isinstance(Node2, function_estimation_node) and isinstance(Node1, function_estimation_node)), \
                ("Error. Invalid type of node {}".format(type(Node2)))

            self.input_index = None
            self.input = None

            self.mutation_method = self.mutate_operation_node
            self.mutation_swap_method = self.mutate_swap_operation_node
            self.operation = operation_node
            self.result_calculation = self.get_result_operation_node
            self.node1 = Node1
            self.node2 = Node2
            self.get_sub_tree = self.get_sub_tree_operation_node
            self.value = 0
            self.update_value()


    def mutate_input_node(self, node_index, input_index, input_value):
        """
        Method to mutate a node, into an operation node. This node can only be an input node.
        Args:
            node_index (int): index of node in the tree (not used in the part, because this is a end condition
            in a recursive algorithm)
            input_index (int):
            input_value (float):

        Returns: number of nodes added to the tree
            int
        """
        self.type_node = 2
        self.node1 = function_estimation_node(self.input_index, self.input, n_inputs=self.number_inputs)
        self.node2 = function_estimation_node(input_index, input_value, n_inputs=self.number_inputs)
        self.value = 3
        self.input_index = None
        self.input = None
        self.operation = self.get_random_operation()
        self.result_calculation = self.get_result_operation_node
        self.get_sub_tree = self.get_sub_tree_operation_node
        self.mutation_method = self.mutate_operation_node
        self.mutation_swap_method = self.mutate_swap_operation_node
        return 2


    def get_random_operation(self):
        """
        Method to get a random operation
        Returns:
            operation
        """
        random_value = random.uniform(0, 1)
        if random_value<0.25:
            return sum_op_function_estimation()
        elif random_value<0.5:
            return rest_op_function_estimation()
        elif random_value<0.75:
            return mult_op_function_estimation()
        else:
            return div_op_function_estimation()

    def copy(self):
        """
        Method to copy this node.

        Returns:
            function_estimation_node
        """
        if self.type_node == 1:
            return function_estimation_node(self.input_index,self.input.copy(), n_inputs=self.number_inputs)
        elif self.type_node == 0:
            return function_estimation_node(None, None, type_node=0)
        else:
            node_son1 = self.node1.copy()
            node_son2 = self.node2.copy()
            new_node = function_estimation_node(None, None, type_node=2, n_inputs=self.number_inputs, operation_node= self.operation.copy(),
                                                Node1=node_son1, Node2=node_son2)
            return new_node

    def get_result_operation_node(self):
        """
        Method to calculate the result of this node. Used only for operation nodes.
        Returns:
            float
        """
        if self.type_node==1:
            print("Error invalid condition meet. type node == 1 in operation node. ")
            return self.input
        elif self.node1.type_node == 0:
            return self.operation([self.operation.get_null_value()]*self.number_inputs,self.node2.calculate_result())
        elif self.node2.type_node == 0:
            return self.operation(self.node1.calculate_result(),[self.operation.get_null_value()]*self.number_inputs)
        return self.operation(self.node1.calculate_result(),self.node2.calculate_result())


    def report(self):

        if self.type_node == 1:
            if self.is_constant():
                return str(self.input[0])
            else:
                return 'x'

        elif self.type_node == 2:
            report1 = self.node1.report()
            report2 = self.node2.report()
            if report1 == "" or report1 == "() ":
                if report2 == "" or report2 == "() ":
                    return "() "
                else:
                    operation_result = self.operation.report()
                    if operation_result == '-':
                        return "({:s}{:s}) ".format(operation_result,
                                                          self.node2.report())
                    elif operation_result == '/':
                        return "({:d} {:s} {:s}) ".format(self.operation.get_null_value(), operation_result,
                                                          self.node2.report())
                    return report2
            else:
                if report2 == "" or report2 == "() ":
                    return report1
                else:
                    return "({:s} {:s} {:s}) ".format(self.node1.report(),self.operation.report(),self.node2.report())
        else:
            return ""

    def is_constant(self):
        """
        Method to check if this input node, contains an input variable x or a constant.
        Returns:
            bool
        """
        value = self.input[0]
        for i in self.input:
            if value != i:
                return False
        return True

    def get_result_input_node(self):
        """
        Method to calculate the result of this node. Used only for input nodes.
        Returns:
            float
        """
        if self.input_index<13:
            if not self.is_constant():
                print("error invalid condition meet. Constant input node is not constant")
        return self.input


class des_chiffres_root_node:
    back_up_inputs: List[int]
    remaining_indexes_inputs: List[int]
    desired_output: float
    __number_inputs: int
    __fitness: int
    remaining_inputs: List[int]

    def __init__(self, inputs, desired_value, input_index=0, type_node = 1,**kwargs):
        """
        Class that serves as an individual to perform mutation and cross-over. This class contains the tree that
        defines a solution for a des chiffres et des lettres problem.

        Args:
            inputs (List(int)):
            desired_value (float):
            input_index (int):
            type_node (int):
            **kwargs: key words parameters used to create new individuals.
        """
        if input_index is None:
            assert type_node==2, "Invalid combination of type node = {:d} and input index {}".format(type_node,
                                                                                                     input_index)
            self.root_node = des_chiffres_node(input_index=None, input_value=None,
                                               type_node=type_node, **kwargs)
        else:
            self.root_node = des_chiffres_node(input_index=input_index,input_value=inputs[input_index],
                                           type_node=type_node,**kwargs)

        self.__number_inputs = len(inputs)
        self.remaining_inputs = inputs
        self.back_up_inputs = inputs.copy()
        self.remaining_indexes_inputs = [i for i in range(len(inputs))]
        self.remaining_inputs.pop(input_index)
        self.remaining_indexes_inputs.pop(input_index)
        self.__fitness = -10000
        self.desired_output = desired_value

    def copy(self):
        """
        Method to copy a root node and its tree.
        Returns:
            des_chiffres_root_node:
        """
        if self.root_node.type_node == 1:
            return des_chiffres_root_node(self.back_up_inputs.copy(),self.desired_output,self.root_node.input_index)
        elif self.root_node.type_node == 2:
            node_son1 = self.root_node.node1.copy()
            node_son2 = self.root_node.node2.copy()
            new_node = des_chiffres_root_node(self.back_up_inputs.copy(),self.desired_output,0,
                                              type_node=2,operation_node = self.root_node.operation.copy(),Node1=node_son1,Node2=node_son2)
            new_node.remaining_inputs = self.remaining_inputs.copy()
            # new_node.set_fitness(self.__fitness)
            new_node.remaining_indexes_inputs = self.remaining_indexes_inputs.copy()

            return new_node
        else:
           print("Error. Invalid type of node", self.type_node)

    def mutation(self):
        """
        Method to perform a mutation in this individual. The mutation affects the root node and therefore the tree
        inside.
        Returns:
            None
        """
        number_nodes = self.root_node.get_value()
        if len(self.remaining_inputs) > 0:

            random_node = random.randint(1,number_nodes)
            random_index = random.randint(0,len(self.remaining_inputs)-1)
            result = self.root_node.mutate_remaining_inputs(random_node, self.remaining_indexes_inputs[random_index], self.remaining_inputs[random_index])
            # self.sum_value()
            if result > 0:
                self.remaining_inputs.pop(random_index)
                self.remaining_indexes_inputs.pop(random_index)

        else:
            random_index1 = random.randint(0, self.__number_inputs-1)
            random_index2 = random.randint(0, self.__number_inputs-1)

            _ = self.root_node.mutation_swap_method([random_index1, random_index2],
                                               [self.back_up_inputs[random_index1],
                                                self.back_up_inputs[random_index2]])

    def cross_over(self,tree_node):
        """
        Cross-over operation done over this individual and a second parent.
        Args:
            tree_node (des_chiffres_root_node):

        Returns:
            (des_chiffres_root_node):
        """
        if tree_node.root_node.get_value()>5*len(self.back_up_inputs):  # Safeguard against excessively large nodes.
            #  its like saying you are not good enough if you are too big.
            random_index = random.randint(0, len(self.back_up_inputs) - 1)
            new_node = des_chiffres_root_node(self.back_up_inputs.copy(), self.desired_output,
                                              input_index=random_index)
            if self.root_node.get_value()>tree_node.root_node.get_value():

                return tree_node.cross_over(new_node)
            # elif self.root_node.get_value()>5*len(self.back_up_inputs):  # still too big need to be reduced by a smaller
            #     # tree
            #     return self.cross_over(new_node)
            # this condition, though logical, its unnecessary since the outcome is the same for this elif and this else
            else: # the current tree is not too big, so cross-over is done as usual.
                return self.cross_over(new_node)


        random_node_to_be_replace = random.randint(1,self.root_node.get_value())
        random_node_for_replacement = random.randint(1,tree_node.root_node.get_value())
        tree_offpsring = self.copy()
        sub_tree_donor = tree_node.root_node.get_sub_tree(random_node_for_replacement)
        indexes_inputs_donor = sub_tree_donor.get_indexes_in_tree([])
        sub_tree_base = self.root_node.get_sub_tree(random_node_to_be_replace)
        indexes_inputs_sub_base = sub_tree_base.get_indexes_in_tree([])



        if tree_offpsring.root_node.is_node(random_node_to_be_replace):
            tree_offpsring.root_node = sub_tree_donor



        else:
            tree_offpsring.root_node.replace_branch(random_node_to_be_replace, sub_tree_donor,indexes_inputs_donor)



        remaining_indexes_input = tree_offpsring.remaining_indexes_inputs
        remaining_inputs = tree_offpsring.remaining_inputs

        for i in indexes_inputs_sub_base:
            remaining_indexes_input += [i]
            remaining_inputs += [self.back_up_inputs[i]]

        for input_index_donor in indexes_inputs_donor:
            for i,index_remaining in enumerate(remaining_indexes_input):
                if index_remaining==input_index_donor:
                    remaining_indexes_input.pop(i)
                    remaining_inputs.pop(i)
                    break




        tree_offpsring.remaining_indexes_inputs = remaining_indexes_input
        tree_offpsring.remaining_inputs = remaining_inputs
        tree_offpsring.root_node.fix_null_nodes()


        return tree_offpsring



    def fitness(self):
        """
        Method that calculate the fitness of this individual by evaluating the absolute error of the result and
        the desired value.
        Returns:

        """
        result = self.root_node.calculate_result()
        self.__fitness = -abs(result-self.desired_output)
        if self.__fitness < -abs(self.desired_output):
            self.__fitness = -abs(self.desired_output)

    def set_fitness(self, fitness):
        self.__fitness = fitness

    def get_fitness(self):
        return self.__fitness

    def report(self):
        """
        Method to get an expression of the solution.
        Returns:
            str

        """
        return self.root_node.report()




class function_estimation_root_node:


    back_up_inputs: List[List[int]]
    __fitness: float
    remaining_inputs: List[List[int]]

    def __init__(self, inputs, desired_value, input_index=0, type_node=1, **kwargs):
        """
        Class that serves as an individual to perform mutation and cross-over. This class contains the tree that
        defines a solution for a des chiffres et des lettres problem.

        Args:
            inputs (List(List(int))): List of all inputs.
            desired_value (List(float)): list of expected outputs.
            input_index (int): index of the input asigned to the root node (valid only of root node is an input node)
            type_node (int): type of node
            **kwargs: key words parameters used to create new individuals.
        """
        data_numbers = len(inputs[0])
        if input_index is None:
            assert type_node==2, "Invalid combination of type node = {:d} and input index {}".format(type_node,
                                                                                                     input_index)

            self.root_node = function_estimation_node(input_index=None, input_value=None,
                                               type_node=type_node,  n_inputs=data_numbers, **kwargs)
        else:
            self.root_node = function_estimation_node(input_index=input_index,input_value=inputs[input_index],
                                           type_node=type_node, n_inputs=data_numbers,**kwargs)

        self.__number_inputs = len(inputs)
        self.remaining_inputs = inputs
        self.back_up_inputs = inputs.copy()
        self.remaining_indexes_inputs = [i for i in range(len(inputs))]
        self.remaining_inputs.pop(input_index)
        self.remaining_indexes_inputs.pop(input_index)
        self.__fitness = -10000
        self.desired_output = desired_value

    def copy(self):
        """
        Method to give a copy of this individual
        Returns:
            function_estimation_root_node:
        """
        if self.root_node.type_node == 1:
            return function_estimation_root_node(self.back_up_inputs.copy(),self.desired_output,self.root_node.input_index)
        elif self.root_node.type_node == 2:
            node_son1 = self.root_node.node1.copy()
            node_son2 = self.root_node.node2.copy()
            new_node = function_estimation_root_node(self.back_up_inputs.copy(),self.desired_output,0,
                                              type_node=2,operation_node = self.root_node.operation.copy(),Node1=node_son1,Node2=node_son2)
            new_node.remaining_inputs = self.remaining_inputs.copy()
            # new_node.set_fitness(self.__fitness)
            new_node.remaining_indexes_inputs = self.remaining_indexes_inputs.copy()

            return new_node
        else:
           print("Error. Invalid type of node :", self.type_node)  ## Delete this one and the one on des_chiffres

    def mutation(self):
        """
        Method to perform a mutation in this individual. The mutation affects the root node and therefore the tree
        inside.
        Returns:
            None
        """
        number_nodes = self.root_node.get_value()
        if len(self.remaining_inputs) > 0:
            random_node = random.randint(1,number_nodes)
            random_index = random.randint(0,len(self.remaining_inputs)-1)
            result = self.root_node.mutate_remaining_inputs(random_node, self.remaining_indexes_inputs[random_index], self.remaining_inputs[random_index].copy())
            # self.sum_value()
            if result > 0:
                self.remaining_inputs.pop(random_index)
                self.remaining_indexes_inputs.pop(random_index)
        else:
            random_index1 = random.randint(0, self.__number_inputs-1)
            random_index2 = random.randint(0, self.__number_inputs-1)

            _ = self.root_node.mutation_swap_method([random_index1, random_index2],
                                               [self.back_up_inputs[random_index1],
                                                self.back_up_inputs[random_index2]])
            if _ != 2:
                print("Error. Impossible condition achieved")

    def cross_over(self,tree_node):
        """
        Cross-over operation done over this individual and a second parent.
        Args:
            tree_node (function_estimation_root_node):

        Returns:
            (function_estimation_root_node):
        """
        if tree_node.root_node.get_value() > 100:  # Safeguard against excessively large nodes.
            #  its like saying you are not good enough if you are too big.
            random_index = random.randint(0, len(self.back_up_inputs) - 1)
            new_node = function_estimation_root_node(self.back_up_inputs.copy(), self.desired_output.copy(),
                                                     input_index=random_index)
            if self.root_node.get_value() > tree_node.root_node.get_value():
                return tree_node.cross_over(new_node)
            # elif self.root_node.get_value()>100:  # still too big need to be reduced by a smaller
            #     # tree
            #     return self.cross_over(new_node)
            # this condition, though logical, its unnecessary since the outcome is the same for this elif and this else
            else:  # the current tree is not too big, so cross-over is done as usual.
                return self.cross_over(new_node)

        random_node_to_be_replace = random.randint(1,self.root_node.get_value())
        random_node_for_replacement = random.randint(1,tree_node.root_node.get_value())
        tree_offpsring = self.copy()
        sub_tree_donor = tree_node.root_node.get_sub_tree(random_node_for_replacement)
        indexes_inputs_donor = sub_tree_donor.get_indexes_in_tree([])
        sub_tree_base = self.root_node.get_sub_tree(random_node_to_be_replace)
        indexes_inputs_sub_base = sub_tree_base.get_indexes_in_tree([])

        if tree_offpsring.root_node.is_node(random_node_to_be_replace):
            tree_offpsring.root_node = sub_tree_donor

        else:
            tree_offpsring.root_node.replace_branch(random_node_to_be_replace, sub_tree_donor,indexes_inputs_donor)

        remaining_indexes_input = tree_offpsring.remaining_indexes_inputs
        remaining_inputs = tree_offpsring.remaining_inputs

        for i in indexes_inputs_sub_base:
            remaining_indexes_input += [i]
            remaining_inputs += [self.back_up_inputs[i].copy()]

        for input_index_donor in indexes_inputs_donor:
            for i,index_remaining in enumerate(remaining_indexes_input):
                if index_remaining==input_index_donor:
                    remaining_indexes_input.pop(i)
                    remaining_inputs.pop(i)
                    break

        tree_offpsring.remaining_indexes_inputs = remaining_indexes_input
        tree_offpsring.remaining_inputs = remaining_inputs
        tree_offpsring.root_node.fix_null_nodes()

        return tree_offpsring



    def fitness(self):
        """
        Method that calculate the fitness of this individual by evaluating the absolute error of the result and
        the desired value.
        Returns:
            None
        """
        result_tree = self.root_node.calculate_result()
        results_per_output = []
        # results = self.root_node.get_result_operation_node()
        for i in range(len(self.desired_output)):
            if self.desired_output[i]!= 0:
                results_per_output += [-(abs((result_tree[i]-self.desired_output[i])/self.desired_output[i]))]
            else:
                results_per_output += [-(abs((result_tree[i]-self.desired_output[i])))]
        self.__fitness = sum(results_per_output)
        if self.__fitness < -100:
            self.__fitness = -100


    def set_fitness(self, fitness):
        self.__fitness = fitness

    def get_fitness(self):
        return self.__fitness

    def report(self):
        """
        Method to get an expression of the operation
        Returns:

        """
        return self.root_node.report()


class operation:
    def __init__(self):
        """
        Base class for calculating and operation
        """
        pass

    def get_null_value(self):
        return 0

    def report(self):
        return "o"

    def copy(self):
        return None

class sum_op(operation):
    def __call__(self, *args):
        return args[0] + args[1]

    def report(self):
        return "+"

    def copy(self):
        return sum_op()

class rest_op(operation):
    def __call__(self, *args):
        return args[0]-args[1]

    def report(self):
        return "-"

    def copy(self):
        return rest_op()

class mult_op(operation):
    def __call__(self, *args):
        return args[0]*args[1]

    def report(self):
        return "*"


    def copy(self):
        return mult_op()
    def get_null_value(self):
        return 1

class div_op(operation):
    def __call__(self, *args):
        if args[1]==0:
            return args[0]*10000
        return args[0]/args[1]

    def get_null_value(self):
        return 1

    def report(self):
        return "/"

    def copy(self):
        return div_op()

class sum_op_function_estimation(sum_op):
    def __call__(self, *args):
        return [args[0][i]+args[1][i] for i in range(len(args[0]))]

    def copy(self):
        return sum_op_function_estimation()

class rest_op_function_estimation(rest_op):
    def __call__(self, *args):
        return [args[0][i]-args[1][i] for i in range(len(args[0]))]
    def copy(self):
        return rest_op_function_estimation()

class mult_op_function_estimation(mult_op):
    def __call__(self, *args):
        return [args[0][i]*args[1][i] for i in range(len(args[0]))]

    def copy(self):
        return mult_op_function_estimation()

class div_op_function_estimation(div_op):

    def __call__(self, *args):
        division = []
        for i in range(len(args[0])):
            if args[1][i]==0:
                division += [args[0][i]*1000000]
            else:
                division += [args[0][i]/args[1][i]]
        return division

    def copy(self):
        return div_op_function_estimation()