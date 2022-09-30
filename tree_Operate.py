from treelib import Tree
DROOT = 0

class Tree_Node(object):
    def __init__(self,dep='',pos='',dep_id=-1):
        self.dep = dep
        self.pos = pos
        self.dep_id = dep_id
        self.cnodes = []


class Phrase_Node(object):
    def __init__(self):
        self.word = ''
        self.word_id = -1

class PhraseTree(object):
    def __init__(self):
        self.phrase_tree = Tree()
    def node_sort(self,node):
        return node.identifier
    def get_all_node(self,cur_node,cnode_list):
        if not cur_node:
            return cnode_list
        child_nodes = self.phrase_tree.children(cur_node.identifier)
        for child_node in child_nodes:
            self.get_all_node(child_node,cnode_list)
            cnode_list.append(child_node)

class STree(object):
    def __init__(self):
        self.dp_tree = Tree()
        self.dp_tree.create_node('DROOT', DROOT, data=Tree_Node())

    def node_sort(self,node):
        return node.identifier

    def get_all_node(self,cur_node,cnode_list):
        if not cur_node:
            return cnode_list
        child_nodes = self.dp_tree.children(cur_node.identifier)
        for child_node in child_nodes:
            self.get_all_node(child_node,cnode_list)
            cnode_list.append(child_node)

    def build_dp_tree(self, words, postags,deps,pnode_id):
        for i,dep in enumerate(deps):
            if deps[i][1] == pnode_id:
                self.dp_tree.create_node(words[i], deps[i][2], parent=pnode_id,
                                         data=Tree_Node(pos=postags[i], dep=deps[i][0], dep_id=pnode_id))
                self.build_dp_tree(words,postags,deps,deps[i][2])

    def build_dp_tree_ltp4(self, words, postags, deps, pnode_id):
        for i, dep in enumerate(deps):
            if deps[i][1] == pnode_id:
                self.dp_tree.create_node(words[i], deps[i][0], parent=pnode_id,
                                      data=Tree_Node(pos=postags[i], dep=deps[i][2]))
                self.build_dp_tree_ltp4(words, postags, deps, deps[i][0])


def build_tree(sour_tree,tgt_tree,tgt_node_ids,tgt,pnode_id):
    for i,node_id in enumerate(tgt_node_ids.keys()):
        if node_id == DROOT:
            continue
        node = sour_tree.dp_tree.get_node(node_id)
        pnode = sour_tree.dp_tree.parent(node_id)
        if pnode.identifier == pnode_id:
            tgt_tree.dp_tree.create_node(node.tag, tgt_node_ids[node_id], parent=tgt_node_ids[pnode.identifier],
                                                data=Tree_Node(pos=node.data.pos, dep=node.data.dep))
            if node_id in tgt.keys():
                del tgt[node_id]
            build_tree(sour_tree,tgt_tree,tgt_node_ids,tgt,node.identifier)

def get_effective_pnode(tree,cur_node,pnode_id_list,tgt_tree):
    nodes = tree.dp_tree.all_nodes()
    for node in nodes:
        if node.identifier == DROOT:
            continue
        pnode = tree.dp_tree.parent(cur_node.identifier)
        if pnode.identifier in pnode_id_list.keys() and tgt_tree.dp_tree.contains(pnode_id_list[pnode.identifier]):
            return pnode
        if pnode.identifier not in pnode_id_list.keys():
            pnode = get_effective_pnode(tree,pnode,pnode_id_list,tgt_tree)
            return pnode
    return tree.dp_tree.children(DROOT)[0]