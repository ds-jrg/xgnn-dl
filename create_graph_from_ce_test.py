from create_graph_from_ce import PyGfromCE
import unittest
import torch
import torch_geometric
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer

class_3 = OWLClass('#3')
class_2 = OWLClass('#2')
class_1 = OWLClass('#1')
class_0 = OWLClass('#0')
edge = OWLObjectProperty('#to')


class TestMutateCE(unittest.TestCase):
    def setUp(self):
        self.new_edge = OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)
        self.ce_3_to_1 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)])
        self.ce_3_to_1OR2 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=OWLObjectUnionOf([class_1, class_2]), property=edge)])

    def testPyGfromCE(self):
        ce = self.ce_3_to_1
        ce_pyg = PyGfromCE()
        graph = ce_pyg.create_pyg_from_ce(ce)
        print(graph)
        self.assertIsInstance(ce_pyg.graph, torch_geometric.data.HeteroData)


if __name__ == '__main__':
    unittest.main()
