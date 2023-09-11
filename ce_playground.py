import numpy
import sys
import copy
import random
import torch
#sys.path.append('/Ontolearn')
#import generatingXgraphs


from ontolearn.concept_learner import CELOE
from ontolearn.model_adapter import ModelAdapter
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer
from examples.experiments_standard import ClosedWorld_ReasonerFactory
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric



dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns

class_3 = OWLClass(IRI(NS,'3'))
class_2 = OWLClass(IRI(NS,'2'))
class_1 = OWLClass(IRI(NS,'1'))
class_0 = OWLClass(IRI(NS,'0'))
edge = OWLObjectProperty(IRI(NS, 'to'))
#CE 3-2-1
edge_end = OWLObjectSomeValuesFrom(property=edge, filler=class_1)
filler_end = OWLObjectIntersectionOf([class_2, edge_end])
edge_middle = OWLObjectSomeValuesFrom(property=edge, filler=filler_end)
ce_321 = OWLObjectIntersectionOf([class_3, edge_middle])


ce_01 = OWLObjectIntersectionOf([class_0, class_1])
print(45, ce_01, ce_01._operands, type(ce_01.operands))    # <class 'method'>


new_operands = list(ce_01._operands)
print(51, new_operands)
new_operands.append(class_0)           # Append an element to the list
ce_01._operands = tuple(new_operands) 
print(ce_01)

print(dlsr.render(ce_01))