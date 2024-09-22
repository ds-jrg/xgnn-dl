import torch
import owlapy
from owlapy import *
from torch_geometric.data import HeteroData
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
from evaluation import InstanceChecker
import unittest


# create a teset hdata graph

data = HeteroData()
data['A'].x = torch.randn(3, 1)
data['B'].x = torch.randn(3, 1)

edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
edge_name = ('A', 'to', 'B')
edge_reverse = ('B', 'to', 'A')
edges_reverse = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]])
data[edge_name].edge_index = edges
data[edge_reverse].edge_index = edges_reverse


class_3 = OWLClass('#3')
class_2 = OWLClass('#2')
class_1 = OWLClass('#1')
class_0 = OWLClass('#0')
edge = OWLObjectProperty('#to')

edge = OWLObjectMinCardinality(
    cardinality=1, filler=class_1, property=edge)


class TestMutateCE(unittest.TestCase):
    def setUp(self):
        data = HeteroData()
        data['1'].x = torch.randn(3, 1)
        data['2'].x = torch.randn(3, 1)

        edges_12 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        edge_name_12 = ('1', 'to', '2')
        edge_reverse_21 = ('2', 'to', '1')
        edges_reverse_21 = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]])
        data[edge_name_12].edge_index = edges_12
        data[edge_reverse_21].edge_index = edges_reverse_21
        edges_13 = torch.tensor([[0, 1, 1, 3], [1, 0, 2, 0]])
        edges_31 = torch.tensor([[1, 0, 2, 0], [0, 1, 1, 3]])
        edge_name_13 = ('1', 'to', '3')
        edge_name_31 = ('3', 'to', '1')
        data[edge_name_13].edge_index = edges_13
        data[edge_name_31].edge_index = edges_31

        self.hdata = data

        # ce
        class_3 = OWLClass('#3')
        class_2 = OWLClass('#2')
        class_1 = OWLClass('#1')
        class_0 = OWLClass('#0')
        edge = OWLObjectProperty('#to')

        ce_1_2 = OWLObjectIntersectionOf([class_1, OWLObjectMinCardinality(
            cardinality=1, filler=class_2, property=edge)])
        self.ce_1_2 = ce_1_2
        ce_card2_12 = OWLObjectIntersectionOf([class_1, OWLObjectMinCardinality(
            cardinality=2, filler=class_2, property=edge)])
        self.card2_12 = ce_card2_12

    def test_getadjacentnodes(self):
        ic = InstanceChecker(self.hdata)
        adjacent_nodes = ic.get_adjacent_nodes(
            '1', 1, 'to', set('2'))
        print(adjacent_nodes)
        self.assertDictEqual(adjacent_nodes, {'2': [0, 2]})

    def test_fast_instance_checker_uic(self):
        ce_1_2 = self.ce_1_2
        ic = InstanceChecker(self.hdata)
        true_nodes = ic.fast_instance_checker_uic(ce_1_2)
        print('Result fic', true_nodes)
        self.assertDictEqual(true_nodes, {'1': [0, 1, 2]})

        ce_c2_12 = self.card2_12
        ic = InstanceChecker(self.hdata)
        true_nodes = ic.fast_instance_checker_uic(ce_c2_12)
        print('Result fic c2', true_nodes)
        self.assertDictEqual(true_nodes, {'1': [1]})


if __name__ == '__main__':
    unittest.main()
