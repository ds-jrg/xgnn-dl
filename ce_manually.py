#In this file, we manually create CEs. 
#The goal is, to be able to use this CEs, to create different graphs and receive the GNN-output on those


#Input: List for a Tree [nodetype1, [nodetype2, [nodetype3]]] would be 
#class1 and edge to class2, where this again has an edge to class3

#We say, that each inside-list is of form [nodetype, []], where the latter list may be empty.

from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLNamedIndividual, IRI


dlsr = DLSyntaxObjectRenderer()
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
NS = xmlns
objprop = OWLObjectProperty(IRI(NS, 'to'))

def create_ce_from_inputtree(tree):
    return_ce = OWLObjectIntersectionOf([])
    tce = tuple(return_ce._operands)
    lce = list(tce)
    current_class = OWLClass(IRI(NS, tree[0]))
    lce.append(current_class)
    return_ce._operands = tuple(lce)
    del tree[0]
    for i in tree: 
        child = OWLObjectIntersectionOf([])
        child = create_ce_from_inputtree(i)
        edge = OWLObjectSomeValuesFrom(property = objprop, filler = child)
        tcec = tuple(return_ce._operands)
        lcec = list(tcec)
        lcec.append(edge)
        return_ce._operands = tuple(lcec)
    return return_ce

testtree = ['1', ['11', ['111']], ['2'], ['3']]
print(42, dlsr.render(create_ce_from_inputtree(testtree)))

'''
def create_and_visualize_graphs_to_ce():
    print(0)
    #create graphs

    #visualize graphs



# ---------------- Testing functions
empty_tuple = ()
empty_list = list(empty_tuple)
print(empty_list)  # Output: []


class_1 = OWLClass(IRI(NS, '1'))
return_ce = OWLObjectIntersectionOf([])
print(tuple(return_ce._operands))
ttest = tuple(return_ce._operands)
print(list(ttest))
lttest = list(ttest)
lttest.append(class_1)
print(lttest)
return_ce._operands = tuple(lttest)
print(return_ce._operands)
'''