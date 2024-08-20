# from create_random_ce import mutate_ce
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
import unittest
import sys
import os
from create_random_ce import Mutation, CEUtils
import copy

dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns

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

    def test_mutate_ce(self):
        ce_3_to_1 = copy.deepcopy(self.ce_3_to_1)
        ce_3_to_1OR2 = copy.deepcopy(self.ce_3_to_1OR2)
        mutate = Mutation(list_of_classes=[class_1], list_of_edgetypes=[edge])
        ce_mutated = mutate.mutate_global(ce_3_to_1)
        self.assertIsInstance(ce_mutated, OWLClassExpression)
        ce_mutated = mutate.mutate_global(ce_3_to_1OR2)
        self.assertIsInstance(ce_mutated, OWLClassExpression)
        insert_ce = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)])
        # check class
        is_ce_mutated = CEUtils.replace_nth_class(
            ce_3_to_1, 1, insert_ce)
        self.assertEqual(is_ce_mutated, True)
        # check intersection
        is_ce_mutated = CEUtils.replace_nth_intersection(
            ce_3_to_1OR2, insert_ce, 1)
        self.assertEqual(is_ce_mutated, True)
        # check union
        is_ce_mutated = CEUtils.replace_nth_cl_w_union(
            ce_3_to_1OR2, insert_ce, 1)
        self.assertEqual(is_ce_mutated, True)
        # check cardinality restriction
        is_ce_mutated = CEUtils.increase_nth_existential_restriction(
            ce_3_to_1OR2, 1)
        self.assertEqual(is_ce_mutated, True)

        # Tests if they function correctly

        # class
        ce_3_to_1 = copy.deepcopy(self.ce_3_to_1)
        ce_3_to_1or2 = copy.deepcopy(self.ce_3_to_1OR2)
        is_ce_mutated = CEUtils.replace_nth_class(
            ce=ce_3_to_1, n=1, newpropertyvalue=self.ce_3_to_1)
        print(dlsr.render(ce_3_to_1),
              "in contrast to", dlsr.render(ce_3_to_1))
        ground_truth = OWLObjectIntersectionOf([OWLObjectMinCardinality(cardinality=1, filler=class_1, property=edge),
                                                OWLObjectIntersectionOf([class_3,
                                                                         OWLObjectMinCardinality(cardinality=1, filler=class_1, property=edge)])])
        ground_truth_str = dlsr.render(ground_truth)
        result_str = dlsr.render(ce_3_to_1)
        self.assertEqual(result_str, ground_truth_str)
        # class 2
        ce_3_to_1 = copy.deepcopy(self.ce_3_to_1)
        ce_3_to_1or2 = copy.deepcopy(self.ce_3_to_1OR2)
        is_ce_mutated = CEUtils.replace_nth_class(
            ce=ce_3_to_1, n=2, newpropertyvalue=self.ce_3_to_1)
        print(dlsr.render(ce_3_to_1),
              "in contrast to", dlsr.render(ce_3_to_1))
        ground_truth = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=self.ce_3_to_1, property=edge)])
        ground_truth_str = dlsr.render(ground_truth)
        result_str = dlsr.render(ce_3_to_1)
        self.assertEqual(result_str, ground_truth_str)

        # intersection
        ce_3_to_1 = copy.deepcopy(self.ce_3_to_1)
        ce_3_to_1or2 = copy.deepcopy(self.ce_3_to_1OR2)
        is_ce_mutated = CEUtils.replace_nth_intersection(
            ce=ce_3_to_1, newedge=self.new_edge, n=1)
        print(dlsr.render(ce_3_to_1),
              "is mutated version of", dlsr.render(self.ce_3_to_1))
        ground_truth = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=2, filler=class_1, property=edge)])
        self.assertEqual(dlsr.render(ground_truth), dlsr.render(ce_3_to_1))

        # union
        ce_3_to_1 = copy.deepcopy(self.ce_3_to_1)
        ce_3_to_1or2 = copy.deepcopy(self.ce_3_to_1OR2)
        is_ce_mutated = CEUtils.replace_nth_cl_w_union(
            ce=ce_3_to_1or2, newclass=class_2, n=1)
        ground_truth = OWLObjectIntersectionOf([OWLObjectUnionOf([class_3, class_2]), OWLObjectMinCardinality(
            cardinality=1, filler=OWLObjectUnionOf([class_1, class_2]), property=edge)])
        print(
            f"{dlsr.render(ce_3_to_1or2)} is mutated version of {dlsr.render(self.ce_3_to_1OR2)}")
        # self.assertEqual(dlsr.render(ground_truth), dlsr.render(ce_3_to_1or2))

        # cardinality
        ce_3_to_1 = copy.deepcopy(self.ce_3_to_1)
        ce_3_to_1or2 = copy.deepcopy(self.ce_3_to_1OR2)
        is_ce_mutated = CEUtils.increase_nth_existential_restriction(
            ce=ce_3_to_1, n=1)
        ground_truth = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=2, filler=class_1, property=edge)])
        print(
            f"{dlsr.render(ce_3_to_1)} is mutated version of {dlsr.render(self.ce_3_to_1)}")
        self.assertEqual(dlsr.render(ground_truth), dlsr.render(ce_3_to_1))


class TestCEUtils(unittest.TestCase):
    def setUp(self):
        self.new_edge = OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)
        self.ce_3_to_1 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=class_1, property=edge)])
        self.ce_3_to_1OR2 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1, filler=OWLObjectUnionOf([class_1, class_2]), property=edge)])
        self.ce_321 = OWLObjectIntersectionOf([class_3, OWLObjectMinCardinality(
            cardinality=1,
            filler=OWLObjectMinCardinality(cardinality=1,
                                           filler=class_1,
                                           property=edge),
            property=edge)])

    def test_get_max_depth(self):
        max_depth = CEUtils.get_max_depth(self.ce_3_to_1)
        self.assertEqual(max_depth, 1)
        max_depth = CEUtils.get_max_depth(self.ce_321)
        self.assertEqual(max_depth, 2)

    def test_find_all_poosible_mutations(self):
        possible_mutations = CEUtils.find_all_poosible_mutations(
            self.ce_3_to_1)
        self.assertEqual(len(possible_mutations), 3)
        poss_mut_class = CEUtils.find_all_poosible_mutations(class_2)
        self.assertEqual(len(poss_mut_class), 1)
        poss_mutate_union = CEUtils.find_all_poosible_mutations(
            self.ce_3_to_1OR2)
        self.assertEqual(len(poss_mutate_union), 4)


if __name__ == '__main__':
    unittest.main()
