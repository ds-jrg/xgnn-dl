
import numpy
import sys
#sys.path.append('/Ontolearn')
#import generatingXgraphs


from ontolearn.concept_learner import CELOE
from ontolearn.model_adapter import ModelAdapter
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.render import DLSyntaxObjectRenderer
from examples.experiments_standard import ClosedWorld_ReasonerFactory
from owlapy.model import OWLObjectProperty
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2



#here we load a dataset from the rest of the code






#General idea:
#We have classes <-> node types
    # These classes don't have subclasses (yet)
    # The classes have slots <-> features, where each slot has a predefined space of what can be put in there
    # instances are ontologies with filled slots so-to-speak

#example


file_owl="file://KGs/example_ontology.owl"
xmlns = "http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#"
manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create(file_owl))
iri = IRI('http://example.com/father#', 'child')

#there are no individuals yet
for ind in onto.individuals_in_signature():
    print(ind)

    
#new class <-> nodetype
def new_class(nodetype, manager, onto):
    new_class = OWLClass(IRI(xmlns, nodetype))
    new_class_declaration_axiom = OWLDeclarationAxiom(new_class)
    manager.add_axiom(onto, new_class_declaration_axiom)
    return new_class

#new data property <-> new feature
def new_data_property(feature, manager, onto):
    new_dp = OWLDataProperty(IRI(xmlns, feature))
    new_dp_declaration_axiom = OWLDeclarationAxiom(new_dp)
    manager.add_axiom(onto, new_dp_declaration_axiom)
    return new_dp

    
#new object property <-> edges
def new_object_property(edge, manager, onto):
    new_op = OWLObjectProperty(IRI(xmlns, edge))
    new_op_declaration_axiom = OWLDeclarationAxiom(new_op)
    manager.add_axiom(onto, new_op_declaration_axiom)

new_data_property('Hej', manager, onto)
manager.save_ontology(onto, IRI.create('file:/' + 'test' + '.owl'))
#new object property <-> new edge





#example: set domain and everything into a file
manager_test = OWLOntologyManager_Owlready2()
onto_test = manager_test.load_ontology(IRI.create(file_owl))
Testclass = new_class('Testclass1', manager_test, onto_test) #this si an owl-class, but no OWLClassExpression?



new_dp = OWLDataProperty(IRI(xmlns, 'TestDT1'))
new_dp_declaration_axiom = OWLDeclarationAxiom(new_dp)

#Testdatatypeprop = new_data_property('TestDT1', manager_test, onto_test)
new_dp_domain = OWLDataPropertyDomainAxiom('TestDT1', 'Testclass1')
new_dp_domain_axiom = OWLDeclarationAxiom(new_dp_domain)
new_dp


manager_test.add_axiom(onto_test, new_dp_declaration_axiom)
#manager_test.add_axiom(onto_test, new_dp_domain_axiom)




#new_object_property('hasTestclass2', manager_test, onto_test)
#new_dp_domain = OWLDataPropertyDomainAxiom('TestDT1', 'Testclass1')
#new_dp_domain_axiom = OWLDeclarationAxiom(new_dp_domain)
#print(manager_test)
# manager_test.add_axiom(onto_test, OWLDeclarationAxiom(new_dp_domain))

manager_test.save_ontology(onto_test, IRI.create('file:/' + 'test_04_05' + '.owl'))









#example from alkid put into a function
#data_property = [[classname, list_of_feature_names], ... ]
def data_prop_list_to_onto(data_property, NS = xmlns):
    print('94: Used data_property', data_property)
    
    for dp_tuple in data_property:
        list_all_Er = []
        counter = 0
        xmlstr = OWLDatatype(IRI("http://www.w3.org/2001/XMLSchema#", "string"))
        for dp in range(len(dp_tuple[1])):
            print('101: for loop over tuples', dp, dp_tuple[1])
            r_new = OWLDataProperty(IRI(NS, 'ER_new'+str(counter)))
 #           E_new_name = 'ER_new' + str(counter)
#            E_new_name =  OWLDataSomeValuesFrom(property=r_new, filler=xmlstr) # TODO: E_new_name is actually a unique name for every feature
            list_all_Er.append(OWLDataSomeValuesFrom(property=r_new, filler=xmlstr))
        dp_class = OWLObjectIntersectionOf(list_all_Er)
        print('created class', dp_class)
        dp_class_owl = OWLClass(IRI(NS, dp_tuple[0]))
        manager.add_axiom(onto, OWLDeclarationAxiom(dp_class_owl))
        manager.add_axiom(onto, OWLEquivalentClassesAxiom([dp_class_owl, dp_class]))
        manager.remove_axiom(onto, OWLEquivalentClassesAxiom([dp_class_owl, dp_class]))
        Er_final = OWLDataSomeValuesFrom(OWLDataProperty(IRI(NS, 'hasSomething')), xmlstr)
        final_class = OWLObjectIntersectionOf(list(dp_class.operands()) + [Er_final])
        manager.add_axiom(onto, OWLEquivalentClassesAxiom([dp_class_owl, final_class]))
    manager.save_ontology(ontology=onto, document_iri=IRI.create('file:/test_Alkid_fct.owl'))

# from hdata: save an ontology
def save_ontology_from_hdata(hd, name_to_be_saved):
    manager = OWLOntologyManager_Owlready2()
    onto = manager.load_ontology(IRI.create(file_owl))
    node_types = hd.node_types #classes
    edge_types = hd.edge_types #object properties
    list_node_features = []
    for nt in hd.node_types:
        list_node_features.append([nt, hd[nt].x])
    for nttuple in list_node_features:
        new_class(nttuple[0], manager, onto)
        count_feat = 0
        for feature in nttuple[1]:
            #new_data_property(str(count_feat), nttuple[0])
            new_data_property(str(count_feat), manager, onto)
            count_feat=count_feat+1
    #missing edge types <-> Object properties
    manager.save_ontology(onto, IRI.create('file:/' + 'test_bashapes' + '.owl'))
    



#call fct
data_prop_list_to_onto([['papertrial', ['Feat1', 'Feat2']]])
    
            
    
    
    
#example Alkid 
NS = xmlns
class_paper = OWLClass(IRI(NS, "Paper"))  #is class expression
class_author = OWLClass(IRI(NS,'Author')) #also class expression
inter_ce = OWLObjectIntersectionOf([class_paper, class_author])
print(inter_ce)


r1 = OWLDataProperty(IRI(NS, 'hasTitle'))
r2 = OWLDataProperty(IRI(NS, 'hasFocusArea'))
r3 = OWLDataProperty(IRI(NS, 'hasAbstract'))

xmlstr = OWLDatatype(IRI("http://www.w3.org/2001/XMLSchema#", "string"))
Er1 = OWLDataSomeValuesFrom(property=r1, filler=xmlstr)
Er2 = OWLDataSomeValuesFrom(property=r2, filler=xmlstr)
Er3 = OWLDataSomeValuesFrom(property=r3, filler=xmlstr)

paper = OWLObjectIntersectionOf([Er1, Er2, Er3])

manager.add_axiom(onto, OWLDeclarationAxiom(class_paper))

manager.add_axiom(onto, OWLEquivalentClassesAxiom([class_paper, paper]))

manager.remove_axiom(onto, OWLEquivalentClassesAxiom([class_paper, paper]))
Er4 = OWLDataSomeValuesFrom(OWLDataProperty(IRI(NS, 'hasSomething')), xmlstr)
new_paper = OWLObjectIntersectionOf(list(paper.operands()) + [Er4])
manager.add_axiom(onto, OWLEquivalentClassesAxiom([class_paper, new_paper]))
manager.save_ontology(ontology=onto, document_iri=IRI.create('file:/test_Alkid.owl'))




#new idea: Class expression = some big class
#We want to create some class and then a class expression like below in the example
#print(class_paper.render(class_paper.concept))





#example code from ontolearn to check if everything works
'''
NS = Namespaces('ex', 'http://example.com/father#')

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}

# Only the class of the learning algorithm is specified
model = ModelAdapter(learner_type=CELOE,
                     reasoner_factory=ClosedWorld_ReasonerFactory,
                     path="KGs/father.owl")

model.fit(pos=positive_examples,
          neg=negative_examples)

dlsr = DLSyntaxObjectRenderer()

for desc in model.best_hypotheses(1):
    print(desc)
    print('The result:', dlsr.render(desc.concept), 'has quality', desc.quality)
    
'''