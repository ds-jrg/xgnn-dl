
from evaluation import FidelityEvaluator
import sys
import copy
import random
import torch
from torch_geometric.data import HeteroData
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
            , all edgetypes as edges (properties)
        """
        list_of_edgetypes = hdata.edge_types
        list_of_classes = hdata.node_types
        if isinstance(list_of_classes, list):
            for count, cls_ in enumerate(list_of_classes):
                if not isinstance(cls_, OWLClass):
                    cls_str = str(cls_)
                    list_of_classes[count] = OWLClass('#'+cls_str)
        if isinstance(list_of_edgetypes, list):
            for count, edge in enumerate(list_of_edgetypes):
                if not isinstance(edge, OWLObjectProperty):
                    str_edge = str(edge)
                    list_of_edgetypes[count] = OWLObjectProperty('#'+str_edge)

        return list_of_classes, list_of_edgetypes

    def get_cl_to_explain(hdata):
        """
        This function takes a dataset
        Output: The class that should be explained
        """
        for node_type in hdata.node_types:
            if hasattr(hdata[node_type], 'y'):
                return OWLClass('#'+str(node_type))
        raise Exception("No class found")


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
                 beam_width=2,
                 beam_depth=7,
                 scoring_function=('fidelity', 'acc'),
                 max_depth=None,
                 number_graphs=10,
                 ):
        self.gnn = gnn
        assert isinstance(data, HeteroData)
        self.data = data
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.max_depth = max_depth
        self.scoring_function = scoring_function
        self.number_graphs = number_graphs
        edge_types = [i[1] for i in self.data.edge_types]
        self.Mutation = Mutation(
            list_of_classes=self.data.node_types,
            list_of_edgetypes=edge_types,
            max_depth=self.max_depth,
        )
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
        class_to_expl = BeamHelper.get_cl_to_explain(self.data)
        assert isinstance(class_to_expl, OWLClass)
        beam = [class_to_expl]*self.beam_width
        print(f"Beam search started with {self.beam_depth} rounds")
        for _ in range(self.beam_depth):
            new_beam = copy.deepcopy(beam)
            for ce in beam:
                assert isinstance(ce, OWLClassExpression), ce
                new_ce = self.Mutation.mutate_global(ce)
                assert new_ce != ce
                new_beam.append(new_ce)
            new_beam.sort(key=self.scoring, reverse=True)
            beam = new_beam[:self.beam_width]
            print(f"Round {_+1} of beamsearch is done")
            print(
                f"The best CE is {dlsr.render(beam[0])} with a score of {round(self.scoring(beam[0]),2)}")
        return beam
