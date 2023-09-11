import numpy
import sys
import copy
import random
import torch
import copy
#sys.path.append('/Ontolearn')
#import generatingXgraphs


from ontolearn.concept_learner import CELOE
from ontolearn.model_adapter import ModelAdapter
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer
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




#  ----------- Functions for manipulating CEs
def return_top_intersection(ce):
    print(48, dlsr.render(ce))
    if isinstance(ce, OWLObjectIntersectionOf):
        print(48, dlsr.render(ce))
        return ce
    elif isinstance(ce, OWLObjectUnionOf):
        for op in ce.operands():
            result = return_top_intersection(op)
            if result is not None:
                return result
    #TODO: Add OWLOBjectSomeValuesFrom


def return_bottom_intersection(ce):
    if isinstance(ce, OWLObjectIntersectionOf):
        print(48, dlsr.render(ce))
        result = []
        for op in ce.operands():
            result.append(return_bottom_intersection(op))
        if all(value is None for value in result):
            return ce
        else:
            for op in ce.operands():
                result = return_bottom_intersection(op)
                if result is not None:
                    return result
            
        return ce
    elif isinstance(ce, OWLObjectUnionOf):
        for op in ce.operands():
            result = return_bottom_intersection(op)
            if result is not None:
                return result
    
    


def add_op_to_intersection(ce, new_op):
    print(60, dlsr.render(ce))
    list_of_operands = list(ce._operands)
    list_of_operands.append(new_op)
    ce._operands = tuple(list_of_operands)
    return ce




def add_op_to_intersection_deepcopy(ce: OWLObjectIntersectionOf, new_op):
    ce_new = copy.deepcopy(ce)
    list_of_operands = list(ce_new._operands)
    list_of_operands.append(new_op)
    ce_new._operands = tuple(list_of_operands)
    return ce_new


#function: Add sth to first intersection
def add_ce_to_top_intersect(ce, new_op):
    top_insec = return_top_intersection(ce)
    print(80, dlsr.render(top_insec))
    add_op_to_intersection(top_insec, new_op)
    return ce

def add_ce_to_bottom_intersect(ce, new_op):
    bottom_insec = return_bottom_intersection(ce)
    print(86, dlsr.render(bottom_insec))
    add_op_to_intersection(bottom_insec, new_op)
    return ce
    
    
    
    
   

# Hilfsfunktion: Tiefen aller Leaf-nodes
#Next steps:
#1. Neu strukturieren
#2. Dictionary to Class Expression
#3. Mutinsert 
#4. GenGrow

#fct-name create_broad_ce

#first: Gengrow from deap (with mutinsert)


#dict: to save spot in the tree: add dictionary to each node: Key: Pythonobject, value: Tiefe, parent, ?
    #Problem: Intersection of 2 intersections: Are these different? > save different class objects in dict



#function: Add sth to last intersection


#function: Save all _operands of all intersections, choose one at random and change this one


#deap-mutation functions: mutinsert oder alles in deap umwandeln und zurück

#gengrow, genfull aus evolearner ?



# --------- Declaration of Testing Objects
ce_01 = OWLObjectIntersectionOf([class_0, class_1])
ce_12 = OWLObjectIntersectionOf([class_1, class_2])
print(44, dlsr.render(ce_01))

ce_u_0_i_12 = OWLObjectUnionOf([class_0, ce_12])











# ------------- Testing Phase of functions
ce_012 = add_op_to_intersection(ce_01, class_2)
print(61, dlsr.render(ce_012))
print(dlsr.render(ce_01))


print('Testing intersection of longer CE')
print(113, dlsr.render(ce_u_0_i_12))
add_ce_to_top_intersect(ce_u_0_i_12, class_3)
print(115, dlsr.render(ce_u_0_i_12))

# ------------------ End Testing Phase













