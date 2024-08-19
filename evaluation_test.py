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
        data['A'].x = torch.randn(3, 1)
        data['B'].x = torch.randn(3, 1)

        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        edge_name = ('A', 'to', 'B')
        edge_reverse = ('B', 'to', 'A')
        edges_reverse = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]])
        data[edge_name].edge_index = edges
        data[edge_reverse].edge_index = edges_reverse
        self.hdata = data

    def test_getadjacentnodes(self):
        adjacent_nodes = InstanceChecker.get_adjacent_nodes(
            self.hdata, 'A', 1, 'to', set('B'))
        print(adjacent_nodes)
        self.assertDictEqual(adjacent_nodes, {'B': [0, 2]})


if __name__ == '__main__':
    unittest.main()
