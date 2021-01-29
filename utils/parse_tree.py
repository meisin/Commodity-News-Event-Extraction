"""
Operations on trees (dependency parse tree)
"""

import numpy as np
from collections import defaultdict
import torch
from torch.autograd import Variable

class Tree(object):
    """ Modified from tree.py from 'Graph Convolution over Pruned Dependency Trees for Relation Extraction' """
 
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def tree_to_adj(sent_len, tree, directed=True, self_loop=False):       
    """
    Convert a tree object to a adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    
    if tree is not None:    
        queue = [tree]
        idx = []
        while len(queue) > 0:
            t, queue = queue[0], queue[1:]

            idx += [t.idx]

            for c in t.children:
                ret[t.idx, c.idx] = 1
            queue += t.children

        if not directed:
            ret = ret + ret.T

        if self_loop:
            for i in idx:
                ret[i, i] = 1
    else:
        print('this is None')
    
    return ret

def head_to_tree(head, tokens, len_, prune, trigger_pos, entity_pos):
    """
    Convert a sequence of head indexes (from dependency parse information) into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    if prune < 0:    ### keep the whole dependency tree - no pruning
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        
        # find dependency path
        trigger_pos = [i for i in range(len_) if trigger_pos[i] == 0]
        entity_pos = [i for i in range(len_) if entity_pos[i] == 0]
        
        cas = None

        trigger_ancestors = set(trigger_pos)
        for t in trigger_pos:
            h = head[t]
            tmp = [t]
            while h > 0:
                tmp += [h-1]
                if len(tmp) == len_ :
                    break
                trigger_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)
        
        entity_ancestors = set(entity_pos)
        for e in entity_pos:
            h = head[e]
            tmp = [e]
            while h > 0:
                tmp += [h-1]
                if len(tmp) == len_ :
                    break
                entity_ancestors.add(h-1)
                h = head[h-1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        lca = -1
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = trigger_ancestors.union(entity_ancestors).difference(cas)
        if lca != -1:
            path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:     ## not in path_nodes
                stack = [i]
                if stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4) # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                if nodes[h-1] is not None:
                    assert nodes[h-1] is not None
                    nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    return root
    
def tokens_tree_adjmatrix(words_in_path, head, words, l, prune, trigger_pos, entity_pos, maxlen):    
    """
    Convert tokens to tree structure then to adjacency matrix
    """
    head, words, trigger_pos, entity_pos = head.cpu().numpy(), words.cpu().detach().numpy(), trigger_pos.cpu().numpy(), entity_pos.cpu().numpy()    
    trees = [head_to_tree(head[i], words[i], l[i], prune, trigger_pos[i], entity_pos[i]) for i in range(len(l))]
       
    adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
    adj = np.concatenate(adj, axis=0)
    adj = torch.from_numpy(adj)
    
    ### for investigating / troubleshooting purpose    
    #for i in range(len(l)):
    #    print_words_from_adj(words_in_path[i], adj[i])  

    return Variable(adj.cuda()) 

def print_words_from_adj(words_in_path, adj):
    """
    For printing sub-dependency parse tree
    """
    words = []
    for rows in adj:
        for col_idx, column in enumerate(rows):
            if column == 1:
                words.append(col_idx)
    words = set(words)
    print(words)
    for word_index in words:
        print(tokenizer.convert_ids_to_tokens(words_in_path[word_index]))