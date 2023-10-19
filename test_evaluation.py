from owlready2 import get_ontology, IRIS
from evaluation import find_adjacent_edges, ce_fast_instance_checker
from evaluation import find_adjacent_edges
from torch_geometric.data import HeteroData
import torch


from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer


import logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.WARNING)


dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns


def test_find_adjacent_edges():
    # Create a HeteroData object with a simple graph
    hetero_data = HeteroData()
    hetero_data['Paper', 'r1', 'Author'].edge_index = torch.tensor([[0, 1], [1, 2]])
    hetero_data['Author', 'r1', 'Paper'].edge_index = torch.tensor([[1, 2], [0, 1]])
    hetero_data['Paper', 'r2', 'Author'].edge_index = torch.tensor([[0, 1], [2, 3]])
    hetero_data['Author', 'r3', 'Paper'].edge_index = torch.tensor([[1, 2, 3], [0, 0, 1]])
    # Test finding adjacent edges for a Paper node
    paper_edges = find_adjacent_edges(hetero_data, 'Paper', 0)
    # assert paper_edges == [(0, 1, ('Paper', 'r1', 'Author')), (0, 1, ('Paper', 'r2', 'Author'))]

    # Test finding adjacent edges for an Author node
    author_edges = find_adjacent_edges(hetero_data, 'Author', 1)
    correct_edges = set([(0, 'Paper', ('Author', 'r1', 'Paper')), (0, 'Paper', ('Author', 'r3', 'Paper'))])
    assert author_edges == correct_edges


# ------------- call the tests -----------------
test_find_adjacent_edges()


def test_ce_fast_instance_checker():
    class_3 = OWLClass(IRI(NS, '3'))
    class_2 = OWLClass(IRI(NS, '2'))
    class_1 = OWLClass(IRI(NS, '1'))
    class_0 = OWLClass(IRI(NS, '0'))
    class_paper = OWLClass(IRI(NS, 'Paper'))
    class_author = OWLClass(IRI(NS, 'Author'))
    edge = OWLObjectProperty(IRI(NS, 'to'))
    # CE 3-2-1
    edge_to_1 = OWLObjectSomeValuesFrom(property=edge, filler=class_1)
    edge_to_2 = OWLObjectSomeValuesFrom(property=edge, filler=class_2)
    filler_with_2_edge_1 = OWLObjectIntersectionOf([class_2, edge_to_1, edge_to_2])
    edge_to_author = OWLObjectSomeValuesFrom(property=edge, filler=class_author)
    paper_to_author = OWLObjectIntersectionOf([class_paper, edge_to_author])
    edge_to_paper = OWLObjectSomeValuesFrom(property=edge, filler=class_paper)
    paper_to_paper = OWLObjectIntersectionOf([class_paper, edge_to_paper])
    paper_to_author_to_paper = OWLObjectIntersectionOf([class_paper, edge_to_author, edge_to_paper])
    # Create a HeteroData object with a simple graph
    hetero_data = HeteroData()
    hetero_data['Paper', 'to', 'Author'].edge_index = torch.tensor([[0, 1], [1, 2]])
    hetero_data['Author', 'to', 'Paper'].edge_index = torch.tensor([[1, 2], [0, 1]])
    # hetero_data['Paper', 'to', 'Author'].edge_index = torch.tensor([[0, 1], [2, 3]])
    # hetero_data['Author', 'to', 'Paper'].edge_index = torch.tensor([[1, 2, 3], [0, 0, 1]])

    # Test for a valid CE
    ce = paper_to_author
    valid_nodes = ce_fast_instance_checker(ce, hetero_data, 'Paper', 0)
    result_set = set([('Author', 1)])
    assert valid_nodes == result_set

    # Test for an invalid CE
    ce = paper_to_paper
    valid_nodes = ce_fast_instance_checker(ce, hetero_data, 'Paper', 0)

    # test for a more complicated CE
    ce = paper_to_author_to_paper
    assert valid_nodes == set()


# ------------- call the tests -----------------
test_find_adjacent_edges()
test_ce_fast_instance_checker()
