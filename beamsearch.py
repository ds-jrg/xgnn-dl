
from evaluation import FidelityEvaluator
import sys
import copy
import random
import torch
import copy
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
from create_random_ce import Mutation, CEUtils
dlsr = DLSyntaxObjectRenderer()


class BeamHelper():

    def get_classes_edges_from_data(hdata):
        """ 
        This functions takes a dataset
        Output: All nodetypes as classes
            , all edgetypes as edges
        """
        return [], []

    def get_cl_to_explain(hdata):
        """
        This function takes a dataset
        Output: The class that should be explained
        """
        return '1'  # as a OWLClass


class BeamSearch():
    """
    This class is the summary of Beam Search
    Goal: Find the best explanatory CEs for a dataset and GNN
    Parameters for search:
    - GNN
    - Dataset
    - How many graphs for each CE should be created
    - Beam width: How many CEs are created
    - max_depth: How "deep" the CEs can be
    - Scoring of the CEs:
        Fidelity
        GNN-Max on created graphs
        Regularization:
            - length of CE
    """

    def __init__(self,
                 gnn,
                 data,
                 beam_width,
                 beam_depth,
                 scoring_function,
                 regularization_length,
                 max_depth=None,
                 number_graphs=10,
                 ):
        self.gnn = gnn
        self.data = data
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.max_depth = max_depth
        self.scoring_function = scoring_function
        self.regularization_length = regularization_length
        self.number_graphs = number_graphs

        self.Mutation = Mutation(self.data.node_types,
                                 self.data.edge_types, self.max_depth)

        self.FidelityEvaluation = FidelityEvaluator(self.data, self.gnn)

    def scoring(self, ce):
        """
        This function takes a CE and scores it.
        """
        if self.scoring_function == ('fidelity', 'acc'):
            return self.FidelityEvaluation.score_fid_accuracy(ce)
        else:
            # TODO: Implement
            pass

    def beam_search(self):
        poss_classes, poss_edges = BeamHelper.get_all_classes_from_dataset(
            self.data)
        class_to_expl = BeamHelper.get_class_to_expl(self.data)
        assert isinstance(class_to_expl, OWLClass)
        beam = [class_to_expl]*self.beam_width

        for _ in range(self.beam_depth):
            for ce in beam:
                new_ce = self.Mutation.mutate_ce(ce)
                beam.append(new_ce)
            beam = beam.sort(key=self.scoring)
            beam = beam[:self.beam_width]
        return beam
