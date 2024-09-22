from torch_geometric import HeteroData
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLObjectCardinalityRestriction, OWLObjectMinCardinality
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLCardinalityRestriction, OWLNaryBooleanClassExpression, OWLObjectRestriction
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.render import DLSyntaxObjectRenderer
dlsr = DLSyntaxObjectRenderer()


class CreateHeteroData():
    def __init__(self):
        pass

    @staticmethod
    def create_hetero_data_from_ce(ce):
        hdata = HeteroData()
        node_counts = {}
        current_nodetype = None
        dict_edges = {}  # entries (nt,edge,nt):(1,2)

        def add_edge_to_dictedges(edge_triple, id0, id1):
            pass

        def loop_ce(ce, current_nt_id=(None, 0)):
            if isinstance(ce, OWLClass):
                add_ce_to_nodecounts()
                current_nodetype = str(dlsr.render(ce))
            elif isinstance(ce, OWLObjectIntersectionOf):
                for op in ce.operands():
                    loop_ce(ce)
