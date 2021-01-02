from unittest import TestCase
import Individuals as ind


class test_des_chiffres_nodes(TestCase):

    def setUp(self):
        # indexes = [6,2,7,3,1,9,5,4]
        # self.tree = ind.des_chiffres_root_node(indexes,37,input_index=2)
        pass

    def get_basic_tree(self):
        inputs = [600, 200, 700, 300, 100, 900, 500, 400]
        return ind.des_chiffres_root_node(inputs, 3700, input_index=2)

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
        self.assertEqual(tree.root_node.get_value(), 1)

        tree.mutation()

        # print("get value: ",tree.get_value())
        self.assertEqual(3, tree.root_node.get_value())

    def test_mutate_remaining_inputs(self):
        """
        Checks mechanic of mutation when not all indexes has been assigned
        Returns: None

        """
        tree = self.get_basic_tree()
        self.assertEqual(1, tree.root_node.type_node)
        tree.mutation()
        self.assertEqual(2, tree.root_node.type_node)
        self.assertEqual(3, tree.root_node.get_value())

    def test_mutate_input_node(self):
        """
        Checks that the input node is transform into a operation node. Checks node sons, values and operation
        Returns: None

        """
        tree = self.get_basic_tree()
        tree.mutation()
        self.assertEqual(1, tree.root_node.node1.type_node)
        self.assertTrue(tree.root_node.node1.input != tree.root_node.node2.input)
        value_node1 = tree.root_node.node1.input
        index_node1 = tree.root_node.node1.input_index
        list_remaining_inputs = tree.remaining_inputs
        list_indexes = tree.remaining_indexes_inputs
        result = tree.root_node.mutate_remaining_inputs(1, list_indexes[0], list_remaining_inputs[0])
        self.assertEqual(2, tree.root_node.node1.type_node)
        self.assertEqual(2, result)
        self.assertEqual(value_node1, tree.root_node.node1.node1.input)
        self.assertEqual(list_remaining_inputs[0], tree.root_node.node1.node2.input)
        self.assertEqual(index_node1, tree.root_node.node1.node1.input_index)
        self.assertEqual(list_indexes[0], tree.root_node.node1.node2.input_index)

    def test_mutate_operation_node(self):
        """
        Check if properly works with null, input and operation node.

        Returns: None

        """
        tree = self.get_basic_tree()
        tree.mutation()
        self.assertEqual(1, tree.root_node.node1.type_node)
        list_remaining_inputs = tree.remaining_inputs
        list_indexes = tree.remaining_indexes_inputs
        result = tree.root_node.mutate_remaining_inputs(1, list_indexes[0], list_remaining_inputs[0])

        self.assertEqual(2, tree.root_node.node1.type_node)
        self.assertEqual(2, result)
        self.assertEqual(3, tree.root_node.node1.get_value())

        result = tree.root_node.mutate_remaining_inputs(2, list_indexes[0], list_remaining_inputs[0])

        self.assertEqual(2, tree.root_node.node1.type_node)
        self.assertEqual(0, result)
        self.assertEqual(3, tree.root_node.node1.get_value())

    def test_cross_over1(self):
        tree1 = self.get_basic_tree()
        tree2 = self.get_basic_tree()
        tree1.mutation()
        tree2.mutation()
        node_tree1 = 3
        node_tree2 = 2
        input_index1 = tree2.root_node.node1.input_index
        input_index2 = tree2.root_node.node2.input_index



        base = tree1.copy()
        sub_tree_donor = tree2.root_node.get_sub_tree(node_tree2)
        indexes_inputs_donor = sub_tree_donor.get_indexes_in_tree([])
        sub_tree_base = tree1.root_node.get_sub_tree(node_tree1)
        indexes_inputs_sub_base = sub_tree_base.get_indexes_in_tree([])

        if base.root_node.is_node(node_tree1):
            base.root_node = sub_tree_donor
        else:
            base.root_node.replace_branch(node_tree1, sub_tree_donor, indexes_inputs_donor)

        remaining_inputs = base.remaining_indexes_inputs

        for input_index in indexes_inputs_sub_base:
            remaining_inputs += [input_index]

        for input_index in indexes_inputs_donor:
            for i, index_remaining in enumerate(remaining_inputs):
                if index_remaining == input_index:
                    remaining_inputs.pop(i)
                    break

        base.root_node.fix_null_nodes()

        self.assertEqual(input_index1, base.root_node.node2.node1.input_index)
        self.assertEqual(input_index2, base.root_node.node2.node2.input_index)


    def test_cross_over2(self):
        tree1 = self.get_basic_tree()
        tree2 = self.get_basic_tree()
        tree1.mutation()
        tree2.mutation()
        new_tree = tree1.cross_over(tree2)
        remaining = new_tree.remaining_indexes_inputs
        input_in_tree = new_tree.root_node.get_indexes_in_tree([])
        back_up_indexes = [i for i in range(len(new_tree.back_up_inputs))]
        for input_index in back_up_indexes:
            self.assertTrue((input_index in remaining) != (input_index in input_in_tree))  # logical xor, input can be in only one of this lists


    def test_copy(self):
        """
        Checks a proper copying for different trees.
        Returns:

        """
        tree1 = self.get_basic_tree()
        tree2 = tree1.copy()
        self.assertEqual(1, tree1.root_node.type_node)
        self.assertEqual(1, tree2.root_node.type_node)
        tree1.mutation()
        tree2.mutation()
        self.assertEqual(1, tree1.root_node.node1.type_node)
        self.assertEqual(1, tree2.root_node.node1.type_node)
        self.assertEqual(2, tree1.root_node.type_node)
        self.assertEqual(2, tree2.root_node.type_node)
        self.assertEqual(3, tree1.root_node.get_value())
        self.assertEqual(3, tree2.root_node.get_value())

        list_remaining_inputs = tree1.remaining_inputs
        list_indexes = tree1.remaining_indexes_inputs
        result = tree1.root_node.mutate_remaining_inputs(1, list_indexes[0], list_remaining_inputs[0])

        self.assertEqual(2, tree1.root_node.node1.type_node)
        self.assertEqual(1, tree2.root_node.node1.type_node)
        self.assertEqual(5, tree1.root_node.get_value())
        self.assertEqual(3, tree2.root_node.get_value())

    def test_get_calculate_result(self):
        """
        Calculates a correct result given a list of indexes.

        Returns:
        """
        tree = self.get_basic_tree()
        tree.mutation()
        num1 = tree.root_node.node1.input
        num2 = tree.root_node.node2.input
        tree.root_node.operation = ind.sum_op()
        result = tree.root_node.calculate_result()
        # print("passed 1")
        self.assertEqual(num1 + num2, tree.root_node.operation(num1, num2))
        # print("passed 2")
        result2 = tree.root_node.get_result_operation_node()
        self.assertEqual(num1 + num2, result2)
        # print("passed 3")
        result2 = tree.root_node.result_calculation()
        self.assertEqual(num1 + num2, result2)
        # print("passed 4")
        self.assertEqual(num1 + num2, result)
        # print("passed 5")
        tree.root_node.operation = ind.rest_op()
        result = tree.root_node.calculate_result()
        self.assertEqual(num1 - num2, tree.root_node.operation(num1, num2))
        # print("passed 6")
        self.assertEqual(num1 - num2, result)
        # print("passed 7")
        tree.root_node.operation = ind.mult_op()
        result = tree.root_node.calculate_result()
        self.assertEqual(num1 * num2, result)
        tree.root_node.operation = ind.div_op()
        result = tree.root_node.calculate_result()
        self.assertEqual(num1 / num2, result)


    def test_mutate_by_swapping_inputs(self):
        tree = self.get_small_tree()
        tree.mutation()
        index_node1 = tree.root_node.node1.input_index
        input_node1 = tree.root_node.node1.input
        index_node2 = tree.root_node.node2.input_index
        input_node2 = tree.root_node.node2.input

        inputs_tree = tree.back_up_inputs
        index_swap1 = 0
        index_swap2 = 1
        nodes_founded = tree.root_node.mutation_swap_method([index_swap1, index_swap2],inputs_tree)
        self.assertEqual(2,nodes_founded)

        self.assertEqual(index_node1,tree.root_node.node2.input_index)
        self.assertEqual(input_node1,tree.root_node.node2.input)

        self.assertEqual(index_node2, tree.root_node.node1.input_index)
        self.assertEqual(input_node2, tree.root_node.node1.input)
