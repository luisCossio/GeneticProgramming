from unittest import TestCase
import Individuals as ind


class test_des_chiffres_nodes(TestCase):

    def setUp(self):
        # inputs = [6,2,7,3,1,9,5,4]
        # self.tree = ind.des_chiffres_root_node(inputs,37,input_index=2)
        pass

    def get_basic_tree(self):
        inputs = [6, 2, 7, 3, 1, 9, 5, 4]
        return ind.des_chiffres_root_node(inputs, 37, input_index=2)

    def get_small_tree(self):
        inputs = [6, 2]
        return ind.des_chiffres_root_node(inputs, 8, input_index=0)


    def test_get_value(self):
        """
        Checks for proper value given a tree.
        Returns: None

        """
        tree = self.get_basic_tree()
        # print("get value: ",tree.get_value())
        self.assertEqual(tree.get_value(), 1)

        tree.mutation()

        # print("get value: ",tree.get_value())
        self.assertEqual(3, tree.get_value())

    def test_mutate_remaining_inputs(self):
        """
        Checks mechanic of mutation when not all inputs has been assigned
        Returns: None

        """
        tree = self.get_basic_tree()
        self.assertEqual(1, tree.type_node)
        tree.mutation()
        self.assertEqual(2, tree.type_node)
        self.assertEqual(3, tree.get_value())

    def test_mutate_swapping_inputs(self):
        """
        Checks for mechanic of completion given all indexes were used
        Returns: None

        """
        pass

    def test_mutate_input_node(self):
        """
        Checks that the input node is transform into a operation node. Checks node sons, values and operation
        Returns: None

        """
        tree = self.get_basic_tree()
        tree.mutation()
        self.assertEqual(1, tree.node1.type_node)
        self.assertTrue(tree.node1.input != tree.node2.input)
        value_node1 = tree.node1.input
        index_node1 = tree.node1.input_index
        list_remaining_inputs = tree.remaining_inputs
        list_indexes = tree.indexes_input
        result = tree.mutate_remaining_inputs(1, list_indexes[0], list_remaining_inputs[0])
        self.assertEqual(2, tree.node1.type_node)
        self.assertEqual(2, result)
        self.assertEqual(value_node1, tree.node1.node1.input)
        self.assertEqual(list_remaining_inputs[0], tree.node1.node2.input)
        self.assertEqual(index_node1, tree.node1.node1.input_index)
        self.assertEqual(list_indexes[0], tree.node1.node2.input_index)

    def test_mutate_operation_node(self):
        """
        Check if properly works with null, input and operation node.

        Returns: None

        """
        tree = self.get_basic_tree()
        tree.mutation()
        self.assertEqual(1, tree.node1.type_node)
        list_remaining_inputs = tree.remaining_inputs
        list_indexes = tree.indexes_input
        result = tree.mutate_remaining_inputs(1, list_indexes[0], list_remaining_inputs[0])

        self.assertEqual(2, tree.node1.type_node)
        self.assertEqual(2, result)
        self.assertEqual(3, tree.node1.get_value())

        result = tree.mutate_remaining_inputs(2, list_indexes[0], list_remaining_inputs[0])

        self.assertEqual(2, tree.node1.type_node)
        self.assertEqual(0, result)
        self.assertEqual(3, tree.node1.get_value())

    def test_cross_over(self):
        # Check for different cases. Diferen size. With and without full indexes achieved.
        pass

    def test_copy(self):
        """
        Checks a proper copying for different trees.
        Returns:

        """
        tree1 = self.get_basic_tree()
        tree2 = tree1.copy()
        self.assertEqual(1, tree1.type_node)
        self.assertEqual(1, tree2.type_node)
        tree1.mutation()
        tree2.mutation()
        self.assertEqual(1, tree1.node1.type_node)
        self.assertEqual(1, tree2.node1.type_node)
        self.assertEqual(2, tree1.type_node)
        self.assertEqual(2, tree2.type_node)
        self.assertEqual(3, tree1.get_value())
        self.assertEqual(3, tree2.get_value())

        list_remaining_inputs = tree1.remaining_inputs
        list_indexes = tree1.indexes_input
        result = tree1.mutate_remaining_inputs(1, list_indexes[0], list_remaining_inputs[0])

        self.assertEqual(2, tree1.node1.type_node)
        self.assertEqual(1, tree2.node1.type_node)
        self.assertEqual(5, tree1.get_value())
        self.assertEqual(3, tree2.get_value())

    def test_get_calculate_result(self):
        """
        Calculates a correct result given a list of inputs.

        Returns:
        """
        tree = self.get_basic_tree()
        tree.mutation()
        num1 = tree.node1.input
        num2 = tree.node2.input
        tree.operation = ind.sum_op()
        result = tree.calculate_result()
        # print("passed 1")
        self.assertEqual(num1 + num2, tree.operation(num1, num2))
        # print("passed 2")
        result2 = tree.get_result_operation_node()
        self.assertEqual(num1 + num2, result2)
        # print("passed 3")
        result2 = tree.result_calculation()
        self.assertEqual(num1 + num2, result2)
        # print("passed 4")
        self.assertEqual(num1 + num2, result)
        # print("passed 5")
        tree.operation = ind.rest_op()
        result = tree.calculate_result()
        self.assertEqual(num1 - num2, tree.operation(num1, num2))
        # print("passed 6")
        self.assertEqual(num1 - num2, result)
        # print("passed 7")
        tree.operation = ind.mult_op()
        result = tree.calculate_result()
        self.assertEqual(num1 * num2, result)
        tree.operation = ind.div_op()
        result = tree.calculate_result()
        self.assertEqual(num1 / num2, result)


    def test_mutate_by_swapping_inputs(self):
        tree = self.get_small_tree()
        tree.mutation()
        index_node1 = tree.node1.input_index
        input_node1 = tree.node1.input
        index_node2 = tree.node2.input_index
        input_node2 = tree.node2.input

        inputs_tree = tree.back_up_inputs
        index_swap1 = 0
        index_swap2 = 1
        nodes_founded = tree.mutation_swap_method([index_swap1, index_swap2],inputs_tree)
        self.assertEqual(2,nodes_founded)


        self.assertEqual(index_node1,tree.node2.input_index)
        self.assertEqual(input_node1,tree.node2.input)

        self.assertEqual(index_node2, tree.node1.input_index)
        self.assertEqual(input_node2, tree.node1.input)
