 # ----------- Functions for saving Class expressions in a Tree-list

def find_tree_depth(tree):
    if not isinstance(tree, list):
        return 0
    if len(tree) == 1:
        return 1
    return 1 + max(find_tree_depth(subtree) for subtree in tree[1:])


def find_longest_path(tree):
    if not isinstance(tree, list):
        return [tree], 1, []

    longest_path, max_depth, index_path = [], 0, [] # needs working !!!!
    for subtree in tree[1:]:
        sub_path, sub_depth = find_longest_path(subtree)
        if sub_depth > max_depth:
            longest_path = sub_path
            max_depth = sub_depth

    return [tree[0]] + longest_path, max_depth + 1


    
# not working
def find_all_paths(tree, current_path=[], all_paths=[]):
    if not isinstance(tree, list):
        all_paths.append(current_path + [tree])
        return

    for index, item in enumerate(tree[1:]):
        find_all_paths(item, current_path + [tree[0]], all_paths)
    #return all_paths

    
    
    
def find_farthest_right_leaf(tree):
    end_reached = False
    farthest_path, farthest_leaf = [], tree
    farthest_path_all = []
    while end_reached == False:
        end_reached = True
        for index, item in enumerate(farthest_leaf):
            if isinstance(item, list):
                end_reached = False
                #  fct for longest path from here
                #  add paths to a common list and the
        

    return farthest_path, farthest_leaf

    

def find_last_elementindex_treelist(nested_list):
    if not isinstance(nested_list, list):
        return None
    for index, item in enumerate(nested_list):
        if isinstance(item, list):
            inner_result = find_last_elementindex_treelist(item)
            if inner_result is not None:
                return [index] + inner_result
        else:
            return [index]
    return None
    
    
def remove_last_elementindex_treelist(nested_list):
    current_list = nested_list
    index_path = find_last_elementindex_treelist(nested_list)
    for index in index_path:
        current_list = current_list[index]
    print(380, current_list)
    if isinstance(current_list[-1], list):
        print(382, current_list[-1])
        if len(current_list[-1]) == 0:
            current_list[-1].pop() 
            
    return nested_list
            
    
    
def get_element_from_index(nested_list, index_path):
    if not isinstance(index, list):
        return nested_list[index]
    current_element = nested_list
    for index in index_path:
        current_element = current_element[index]
    return current_element

    

def ce_to_tree_list(ce):
    current_list = []
    if isinstance(ce, OWLClass):
        current_list.append(remove_front(ce.to_string_id()))
        return current_list
    elif isinstance(ce, OWLObjectIntersectionOf):
        current_list.append('Intersection')
        for op in ce.operands():
            current_list.append(ce_to_tree_list(op))
        return current_list
    elif isinstance(ce, OWLObjectProperty):
        current_list.append(remove_front(ce.to_string_id()))
        return current_list
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        current_list.append(remove_front(ce._property.to_string_id()))
        current_list.append(ce_to_tree_list(ce._filler))
        return current_list
    return current_list


def tree_list_to_ce(tree):
    current_filler = ''
    current_intersection_list = []
    current_leaf = ''
    
    
   


 # -------------- Other Functions
    
def find_last_class(ce, last_class = None):
    print('335 ce',dlsr.render(ce))
    if isinstance(ce, OWLObjectIntersectionOf):
        for op in ce.operands():
            if isinstance(op, OWLClass):
                last_class = op
        counter = 0
        for op in ce.operands():
            if isinstance(op, OWLClass):
                pass
            else:
                counter += 1
                return find_last_class(op, last_class)
        if counter == 0:
            return last_class
    if isinstance(ce, OWLObjectSomeValuesFrom):
        if isinstance(ce._filler, OWLClass):
            return ce._filler
        else:
            return find_last_class(ce._filler, last_class)
    if isinstance(ce, OWLObjectProperty):
        return last_class
    if isinstance(ce, OWLClass):
        print('ce 355 should only occur when ce is exactly one class')
        return ce
    print('ce 359', ce)
    return ce, last_class

def add_edge_and_class_to_end_of_ce(ce, objprop, class_to_add):
    ce_last_class = find_last_class(ce)