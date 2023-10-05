import unittest
from unittest.mock import patch

from create_random_ce import mutate_ce
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression, OWLObjectIntersectionOf
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns


class TestMutateCE(unittest.TestCase):

    @patch('random.choice', side_effect=['add_intersection_with_edge_with_class', 'intersection',
                                         OWLClass(IRI(NS, '0')), OWLObjectProperty(IRI(NS, 'to'))])
    @patch('random.randint', side_effect=[4])
    def test_add_intersection_at_first_intersection(self, mock_randint, mock_choice):
        # Define your initial CE, list_of_classes, and list_of_edge_types
        class_3 = OWLClass(IRI(NS, '3'))
        class_2 = OWLClass(IRI(NS, '2'))
        class_1 = OWLClass(IRI(NS, '1'))
        class_0 = OWLClass(IRI(NS, '0'))
        edge = OWLObjectProperty(IRI(NS, 'to'))
        # CE 3-2-1
        edge_to_1 = OWLObjectSomeValuesFrom(property=edge, filler=class_1)
        edge_to_2 = OWLObjectSomeValuesFrom(property=edge, filler=class_2)
        filler_with_2_edge_1 = OWLObjectIntersectionOf([class_2, edge_to_1, edge_to_2])
        edge_to_filler = OWLObjectSomeValuesFrom(property=edge, filler=filler_with_2_edge_1)
        ce_321 = OWLObjectIntersectionOf([class_3, edge_to_filler])
        ce = ce_321
        list_of_classes = [class_0, class_1, class_2, class_3]
        list_of_edge_types = [edge]

        # ce after adding 0 to the 2nd Intersection (id=1)
        # Call the function
        filler_with_2_edge_1_and_intersect0 = OWLObjectIntersectionOf([class_0, filler_with_2_edge_1])
        edge_to_filler_intersection0 = OWLObjectSomeValuesFrom(property=edge,
                                                               filler=filler_with_2_edge_1_and_intersect0)
        expected_ce = OWLObjectIntersectionOf([class_3, edge_to_filler_intersection0])
        new_ce = mutate_ce(ce, list_of_classes, list_of_edge_types)
        print('previous: %s' % dlsr.render(ce))
        print('expected: ', dlsr.render(expected_ce), 'got:', dlsr.render(new_ce))
        self.assertEqual(new_ce, expected_ce)

    def test_mutate_filler_add(self):
        pass
        # Similar test for mutating filler

    def test_no_intersections(self):
        pass
        # Test for handling no intersections

    def test_invalid_intersection_number(self):
        pass
        # Test for invalid intersection number

    def test_invalid_filler_number(self):
        pass
        # Test for invalid filler number


if __name__ == '__main__':
    unittest.main()
