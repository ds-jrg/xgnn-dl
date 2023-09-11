for i in range(5):
    print(i)
'''
# TODO: Delete if it is not used
def create_hdata_graphs_from_ce_list(ce_list, possible_nodes, possible_edges, possible_features):
    print('Graphs from CE are being created')
    
    #create dictionary for meta-graph from available data + randomly complete rest of the data
    # dict has form {(start_node, edge, end_node):(torch.tensor([0], dtype = torch.long), torch.tensor([0], dtype = torch.long))}  #this front and back!!
    #dict: node-types + count, how many nodes of these types have been initiated until now
    node_ids_with_created_graphs = []
    node_types_count = {}
    for nodet in possible_nodes:
        node_types_count[nodet] = 0
    dict_graph = {}
    for _ in range(len(ce_list)-1): #last node not a new start-node, but maybe a end-node
        if ce_list[_][0] in node_ids_with_created_graphs:
            print('this node is already created, just new edges and features have to be added')
            #just the start_node_type has to be adjusted here to the previous found one.
        else:
            if len(ce_list[_][5]) == 0:
                pass
            else:
                start_list = possible_nodes
                end_list = possible_nodes
                edge_list = possible_edges
                if len(ce_list[_][-1]) != 0:
                    start_list = ce_list[_][-1]
                if len(ce_list[_+1][-1]) != 0:
                    end_list = ce_list[_+1][-1]
                if len(ce_list[_+1][3]) != 0:
                    edge_list = ce_list[_+1][3]
                for edge_id in ce_list[_][5]:
                    start_list = possible_nodes
                    end_list = possible_nodes
                    edge_list = possible_edges
                    if len(ce_list[_][-1]) != 0:
                        start_list = ce_list[_][-1]
                    if len(ce_list[edge_id][-1]) != 0:
                        end_list = ce_list[edge_id][-1]
                    if len(ce_list[edge_id][3]) != 0: #edge_type is stored in end-node
                        edge_list = ce_list[edge_id][3]
                    start_node = random.choice(start_list)
                    start_node_id = node_types_count[start_node]
                    node_types_count[start_node] += 1
                    end_node = random.choice(end_list)
                    end_node_id = node_types_count[end_node]
                    node_types_count[end_node] += 1
                    edge_type = random.choice(edge_list)
                    if (start_node, edge_type, end_node) in dict_graph.keys():
                        # extract tensors
                        tensor_start_end = dict_graph[(start_node, edge_type, end_node)]
                        # update tensors
                        tensor_start = tensor_start_end[0]
                        tensor_start = torch.cat((tensor_start, torch.tensor([start_node_id], dtype = torch.long)),0)
                        tensor_end = tensor_start_end[1]
                        tensor_end = torch.cat((tensor_end, torch.tensor([end_node_id], dtype = torch.long)),0)
                    else:
                        # if edge is new, create a tensor for start and end each, just containing the indices of the new edge
                        tensor_start = torch.tensor([start_node_id], dtype = torch.long)
                        tensor_end = torch.tensor([end_node_id], dtype = torch.long)
                    #update used list
                    if start_node == end_node and start_node_id != end_node_id:
                        if (start_node, edge_type, end_node) in dict_graph.keys():
                            tensor_start_end = dict_graph[(start_node, edge_type, end_node)]
                            tensor_start = tensor_start_end[0]
                            tensor_start = torch.cat((tensor_start, torch.tensor([start_node_id, end_node_id], dtype = torch.long)),0)
                            tensor_end = tensor_start_end[1]
                            tensor_end = torch.cat((tensor_end, torch.tensor([end_node_id, start_node_id], dtype = torch.long)),0)
                        else:
                            tensor_start = torch.tensor([start_node_id], dtype = torch.long)
                            tensor_end = torch.tensor([end_node_id], dtype = torch.long)
                        dict_graph.update({(start_node, edge_type, end_node):(tensor_start, tensor_end)})
                    else:
                        dict_graph.update({(start_node, edge_type, end_node):(tensor_start, tensor_end)})
                        dict_graph.update({(end_node, edge_type, start_node):(tensor_end, tensor_start)})
    return dict_graph










#from ce_generation.py
#just for exploration: 
list_nodetypes = [] #strings
def iterate_over_ce(ce: OWLClassExpression):
    
    if isinstance(ce,OWLObjectIntersectionOf):
        list_result = []
        #inter_ce = (OWLObjectIntersectionOf)ce
        operands = ce.operands()
        for op in operands:
            list_result.append(iterate_over_ce(op))
        return list_result
        #list_nodetype.append(operands)  
    elif isinstance(ce, OWLClass):
        #owlclass_ce = (OWLClass)ce
        string_id = ce.to_string_id()
        #print(string_id)
        return [string_id]
    else: 
        print('Error')
        print(type(ce))
        return []
# TODO: Check if this can be deleted
#TODO: Save the links between two node-types
def loop_intersect(ce: OWLClassExpression, list_result, list_of_node_types, list_version = [], total_intersect_count = 0, current_node_id = 0):
    if len(list_result)==0:
        next_node_id=0
    else:
        next_node_id = list_result[-1][0]+1   #current_node_id just for counting the different nodes in the end
    node_id_for_adding_properties = 0 #changes, if an edge property and properties go over this new node from the edge property
    #total_union_count = 0
    
    if isinstance(ce, OWLClass):
        #list_result.append([next_node_id, list_version, total_intersect_count, [], [], remove_front(ce.to_string_id())])
        #next_node_id = next_node_id+1
        #add to current_node_id a OWL-Class to list of classes
        #print(list_result[current_node_id])
        list_result[current_node_id][-1].append(remove_front(ce.to_string_id()))
    
    elif isinstance(ce, OWLObjectIntersectionOf):
        for op in ce.operands():
            loop_intersect(op, list_result, list_of_node_types, list_version, total_intersect_count, current_node_id)
            total_intersect_count = total_intersect_count+1
    elif isinstance(ce, OWLObjectUnionOf):
        #total_union_count = total_union_count+1
        iterate_help = 0 
        for op in ce.operands():
            list_temp = copy.deepcopy(list_version)
            list_temp.append(iterate_help)
            iterate_help = iterate_help+1
            loop_intersect(op, list_result, list_of_node_types, list_temp, total_intersect_count, current_node_id)
    
    #object property
    elif isinstance(ce, OWLObjectProperty):
        #goal 1: retrieve edge and node names / types
        #print(ce.to_string_id())
        #print(list_of_node_types)
        #print('edge_type: ', get_edge_node_types(ce.to_string_id(), list_of_node_types))
        edge_type, endnode_type = get_edge_node_types(ce.to_string_id(), list_of_node_types)
        edge_total = edge_type#+endnode_type
        
        
        
        #WRONGG
        list_result[current_node_id][3].append(edge_total)
        

    #data property
    elif isinstance(ce, OWLDataProperty):
        #test again if this works !!!
        list_result[current_node_id][4].append(remove_front(ce.to_string_id()))
    
    #obj-prop with additional info
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        
        #add empty list
        print('202', list_result[current_node_id])
        list_result[current_node_id][5].append(next_node_id)
        list_result.append([next_node_id, list_version, total_intersect_count, [], [], [], []])
        next_node_id +=1
        current_node_id +=1
        for op in [ce._property, ce._filler]:
            loop_intersect(op, list_result, list_of_node_types, list_version, total_intersect_count, current_node_id)
        current_node_id -=1
        
        #TODO: save in list_result, if the edge goes to a node with node-ID
        
        
        #_property is the edge -> call function again
        #_filler is the end of the edge -> save to be in a new node
    else: 
        print('Error')
        print(type(ce))
    return list_result
            

            
        


def class_expression_to_lists(ce: OWLClassExpression, root_node_type):
    #goal: Use this lists below; but until now, this does not work and other, simpler lists are created
    list_existential = [0, [0], 0, [],[]]
    list_forall = [0, [0], 0, [],[]]
    list_negation = [0, [0], 0, [],[]] 
    
    #may not have to be used
    in_forall_clause = False
    in_existential_clause = True
    in_negation_clause = False
    in_union = False #only true, if in the first side of the union
    
    
    
    list_of_node_types = readout_OWLclass(ce)
    #current_index = -1
    #list_start = [-1]
    list_result = [[0, [], 0, [], [],[], [root_node_type]]]  #versions saved with [1], [10], [11], .... ; intersection-types with accending integers
    #loop through possible start-points:
    #if isinstance(ce, OWLObjectIntersectionOf):
    #    result_list = loop_intersect(ce, result_list, list_version)
    #if isinstance(ce, OWLObjectUnionOf):
    #    result_list = loop_intersect(ce, result_list, list_version)
    print('list of node types:' , list_of_node_types)
    print('list of edge types: not yet implemented')
    result_list = loop_intersect(ce, list_result, list_of_node_types)
    return result_list
'''