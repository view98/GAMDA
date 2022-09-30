from torch import nn
from collections import Counter, defaultdict
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
import scipy.sparse as sp
import linecache
import copy
import logging
import json
import pickle
import os
import gensim
from tree_Operate import *

logger = logging.getLogger(__name__)

labels_lookup = {'Causality':0,'Constituent':1,'Contrast':2,'Degree':3,'Inference':4,'Parataxis':5,'Phenomenon-Instance':6,
                 'Supplement':7,'Temporality':8,'Other':9}

global token_max_len,phrase_max_len,structure_max_len,event_max_len

structure_list = ['ROOT','ccomp','conj','csubj''csubjpass','dobj','iobj','nsubj','nsubjpass','subj','csubj','xsubj','obj','dobj','comp','xcomp',
                  'acomp','obl','dep','cop']

def load_datasets_and_vocabs(args):
    global token_max_len,phrase_max_len,structure_max_len,event_max_len
    token_max_len = args.token_nums
    phrase_max_len = args.phrase_nums
    structure_max_len = args.structure_nums
    event_max_len = args.event_nums
    train_example_file = os.path.join(args.cache_dir, 'train_example.pkl')
    test_example_file = os.path.join(args.cache_dir, 'test_example.pkl')
    train_weight_file = os.path.join(args.cache_dir, 'train_weight_catch.txt')
    test_weight_file = os.path.join(args.cache_dir, 'test_weight_catch.txt')

    if os.path.exists(train_example_file) and os.path.exists(test_example_file):
        logger.info('Loading train_example from %s', train_example_file)
        with open(train_example_file, 'rb') as f:
            train_examples = pickle.load(f)

        logger.info('Loading test_example from %s', test_example_file)
        with open(test_example_file, 'rb') as f:
            test_examples = pickle.load(f)

        with open(train_weight_file, 'rb') as f:
            train_labels_weight = torch.tensor(json.load(f))
        with open(test_weight_file, 'rb') as f:
            test_labels_weight = torch.tensor(json.load(f))

    else:
        train_tree_file = os.path.join(args.dataset_path,'train_relations.pkl')
        test_tree_file = os.path.join(args.dataset_path,'test_relations.pkl')
        logger.info('Loading train trees')
        with open(train_tree_file, 'rb') as f:
            train_trees = pickle.load(f)
        logger.info('Loading test trees')
        with open(test_tree_file, 'rb') as f:
            test_trees = pickle.load(f)

        # get examples of data
        train_examples,train_labels_weight = create_example(train_trees,labels_lookup)
        test_examples,test_labels_weight = create_example(test_trees,labels_lookup)

        logger.info('Creating train examples')
        with open(train_example_file,'wb') as f:
            pickle.dump(train_examples,f,-1)

        logger.info('Creating test examples')
        with open(test_example_file,'wb') as f:
            pickle.dump(test_examples,f,-1)

        with open(train_weight_file,'w') as wf:
            json.dump(train_labels_weight.detach().cpu().numpy().tolist(),wf)
        logger.info('Creating test_weight_cache')
        with open(test_weight_file,'w') as wf:
            json.dump(test_labels_weight.detach().cpu().numpy().tolist(),wf)

    logger.info('Train set size: %s', len(train_examples))
    logger.info('Test set size: %s,', len(test_examples))

    # Build word vocabulary and save pickles.
    word_vecs,word_vocab = load_and_cache_vocabs(train_examples + test_examples,args)

    embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
    if args.embedding_type == 'glove':
        args.glove_embedding = embedding
    else:
        args.word2vec_embedding = embedding

    train_dataset = ED_Dataset(train_examples, args,word_vocab)
    test_dataset = ED_Dataset(test_examples, args,word_vocab)

    return train_dataset,train_labels_weight, test_dataset,test_labels_weight,word_vocab

def create_example(trees,labels_lookup):
    dep_trees,phrase_trees,example_names,labels = trees[0],trees[1],trees[2],trees[3]
    examples = []
    names = []
    label_ids = []
    for i,tree in enumerate(dep_trees):
        tree.dp_tree.show()
        token_edges = []
        example = {'t_ids': [], 'tokens': [], 'pos': [], 'deps': []}
        nodes = tree.dp_tree.all_nodes()
        if len(nodes) == 1:
            continue
        nodes.sort(key=tree.node_sort)
        for node in nodes:
            if node.identifier == DROOT:
                continue

            example['t_ids'].append(node.identifier)
            example['tokens'].append(node.tag)

            # add token edge
            token_edges.append([node.identifier,node.identifier])  # self join
            # parent
            pnode = tree.dp_tree.parent(node.identifier)
            if pnode and pnode.identifier != DROOT:
                token_edges.append([node.identifier,pnode.identifier])
                token_edges.append([pnode.identifier,node.identifier])
            # children
            child_nodes = []
            tree.get_all_node(node,child_nodes)
            for child_node in child_nodes:
                token_edges.append([node.identifier,child_node.identifier])
                token_edges.append([child_node.identifier,node.identifier])

        phrase_t_ids,phrase_edges,token2phrase,phrase_dep_tree = get_phraseEdges_and_convertMatrix(tree,phrase_trees[i])
        structure_t_ids, structure_edges, phrase2structure, structure_dep_tree = get_structureEdges_and_convertMatrix(phrase_dep_tree)
        structure2event = get_event_convertMatrix(structure_dep_tree)

        token_edges = remove_repetion(token_edges)
        token_adj = build_adj(token_edges,example['t_ids'])

        phrase_edges = remove_repetion(phrase_edges)
        phrase_adj = build_adj(phrase_edges,phrase_t_ids)

        structure_edges = remove_repetion(structure_edges)
        structure_adj = build_adj(structure_edges, structure_t_ids)

        example['token_adj'] = token_adj.numpy().tolist()
        example['phrase_adj'] = phrase_adj.numpy().tolist()
        example['structure_adj'] = structure_adj.numpy().tolist()
        example['token2phrase'] = token2phrase.tolist()
        example['phrase2structure'] = phrase2structure.tolist()
        example['structure2event'] = structure2event.tolist()

        example['label'] = labels_lookup[labels[i]]
        examples.append(example)
        label_ids.append(labels_lookup[labels[i]])

        names.append(example_names[i])

    examples = add_events_and_adj(examples,names)

    weight_tensor = get_labels_weight(label_ids,labels_lookup)
    return examples,weight_tensor

def get_labels_weight(labels,labels_lookup):
    label_ids = labels
    nums_labels = Counter(labels)
    nums_labels = [(l,k) for k, l in sorted([(j, i) for i, j in nums_labels.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size//2-1][1])/2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weight_list = []
    for value_id in labels_lookup.values():
        if value_id not in label_ids:
            weight_list.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weight_list.append(median/label[1])
                    break

    weight_tensor = torch.tensor(weight_list,dtype=torch.float32)
    return weight_tensor


def get_phraseEdges_and_convertMatrix(tree,ptree):
    nodes = ptree.phrase_tree.all_nodes()
    combine_ids = []
    nodes.sort(key=ptree.node_sort)
    for node in nodes:
        if node.identifier == DROOT:
            continue
        if node.tag in ['NP','PP']:
            cnodes = []
            ptree.get_all_node(node,cnodes)
            for cnode in cnodes:
                if cnode.is_leaf() and cnode.data.word_id not in combine_ids:
                    combine_ids.append(cnode.data.word_id)
    phrase_node_ids = []
    nodes = tree.dp_tree.all_nodes()
    nodes.sort(key=tree.node_sort)
    for node in nodes:
        if node.identifier == DROOT:
            continue
        if node.identifier in combine_ids:
            cnodes = []
            tree.get_all_node(node, cnodes)
            for cnode in cnodes:
                # add the descendant nodes of node
                if cnode.identifier in combine_ids:
                    node.data.cnodes.append(cnode.identifier)
                    combine_ids.remove(cnode.identifier)
                node.data.cnodes += cnode.data.cnodes
            node.data.cnodes = remove_repetion(node.data.cnodes)
        else:
            phrase_node_ids.append(node.identifier)
    # get all node in phrase dep tree, and ids are in token dep tree
    phrase_node_ids += combine_ids

    # ------------------build phrase dep tree--------------------
    phrase_node_ids.sort()
    phrase_dep_tree = STree()
    #build node id mapping between token2phrase
    tgt_node_ids = {0:0}
    for i,node_id in enumerate(phrase_node_ids):
        tgt_node_ids[node_id] = i + 1

    tgt = copy.deepcopy(tgt_node_ids)
    build_tree(tree, phrase_dep_tree, tgt_node_ids,tgt, DROOT)

    del tgt[DROOT]
    # handle the case where node's parent need be combined, but the node itself is not
    for node_id in tgt.keys():
        node = tree.dp_tree.get_node(node_id)
        # get the effective ancestor node
        effective_pnode = get_effective_pnode(tree,node,tgt_node_ids,phrase_dep_tree)
        phrase_dep_tree.dp_tree.create_node(node.tag, tgt_node_ids[node_id], parent=tgt_node_ids[effective_pnode.identifier],
                                     data=Tree_Node(pos=node.data.pos, dep=node.data.dep, dep_id=effective_pnode.identifier))

    # --------------generate token2phrase array------------------
    phrase_nodes = phrase_dep_tree.dp_tree.all_nodes()
    phrase_nodes.sort(key=tree.node_sort)
    assert len(phrase_nodes) == len(tgt_node_ids)

    del tgt_node_ids[0]
    token2phrase = np.zeros((len(nodes) - 1, len(phrase_nodes) - 1),dtype=np.int)
    for token_node_id,phrase_node_id in tgt_node_ids.items():
        node = tree.dp_tree.get_node(token_node_id)
        token2phrase[token_node_id - 1][phrase_node_id - 1] = 1
        # handle combined nodes
        cnodes = node.data.cnodes
        for node_id in cnodes:
            token2phrase[node_id - 1][phrase_node_id - 1] = 1

    phrase_t_ids, dep_edges = get_dep_edges(phrase_dep_tree)
    return phrase_t_ids,dep_edges,token2phrase,phrase_dep_tree

def get_structureEdges_and_convertMatrix(ptree):
    nodes = ptree.dp_tree.all_nodes()
    remove_node_ids = []
    structure_node_ids = []
    nodes.sort(key=ptree.node_sort)
    for node in nodes:
        if node.identifier == DROOT:
            continue
        if node.data.dep not in structure_list or (node.data.dep == 'dep' and node.data.pos not in ['VB', 'VV']):
            cnodes = []
            ptree.get_all_node(node, cnodes)
            remove_node_ids.append(node.identifier)
            for cnode in cnodes:
                # add the descendant nodes of node
                node.data.cnodes.append(cnode.identifier)
                remove_node_ids.append(cnode.identifier)
                node.data.cnodes += cnode.data.cnodes
            node.data.cnodes = remove_repetion(node.data.cnodes)
            pnode = ptree.dp_tree.parent(node.identifier)
            pnode.data.cnodes += node.data.cnodes
            pnode.data.cnodes.append(node.identifier)
            pnode.data.cnodes = remove_repetion(pnode.data.cnodes)
        else:
            if node.identifier not in remove_node_ids:
                structure_node_ids.append(node.identifier)

    tgt_node_ids = {0: 0}
    structure_node_ids.sort()
    for i, node_id in enumerate(structure_node_ids):
        tgt_node_ids[node_id] = i + 1

    # ptree.dp_tree.show()
    structure_dep_tree = STree()
    tgt = copy.deepcopy(tgt_node_ids)
    build_tree(ptree, structure_dep_tree, tgt_node_ids, tgt, DROOT)

    del tgt[DROOT]
    # handle the case where node's parent need be combined, but the node itself is not
    for node_id in tgt.keys():
        node = ptree.dp_tree.get_node(node_id)
        # get the effective ancestor node
        effective_pnode = get_effective_pnode(ptree, node, tgt_node_ids,structure_dep_tree)
        structure_dep_tree.dp_tree.create_node(node.tag, tgt_node_ids[node_id],
                                            parent=tgt_node_ids[effective_pnode.identifier],
                                            data=Tree_Node(pos=node.data.pos, dep=node.data.dep,
                                                           dep_id=effective_pnode.identifier))

    # --------------generate phrase2structure array------------------
    phrase_nodes = ptree.dp_tree.all_nodes()
    structure_nodes = structure_dep_tree.dp_tree.all_nodes()
    assert len(structure_nodes) == len(tgt_node_ids)
    del tgt_node_ids[0]
    phrase2structure = np.zeros((len(phrase_nodes) - 1, len(structure_nodes) - 1), dtype=np.int)
    for phrase_node_id, structure_node_id in tgt_node_ids.items():
        node = ptree.dp_tree.get_node(phrase_node_id)
        phrase2structure[phrase_node_id - 1][structure_node_id - 1] = 1
        # handle combined nodes
        cnodes = node.data.cnodes
        for node_id in cnodes:
            phrase2structure[node_id - 1][structure_node_id - 1] = 1
    structure_t_ids, structure_edges = get_dep_edges(structure_dep_tree)
    return structure_t_ids,structure_edges,phrase2structure,structure_dep_tree

def get_event_convertMatrix(stree):
    nodes = stree.dp_tree.all_nodes()
    structure2event = np.ones((len(nodes) - 1, 1))
    return structure2event

def add_events_and_adj(examples,names):
    for i,example in enumerate(examples):
        events = []
        for j,e in enumerate(examples):
            if names[i] == names[j]:
                events.append(e['tokens'])
        examples[i]['events'] = events

        event_adj = np.ones((len(events),len(events)),dtype=np.int)
        examples[i]['event_adj'] = event_adj.tolist()

    return examples

def get_dep_edges(ptree):
    dep_edges = []
    t_ids = []
    nodes = ptree.dp_tree.all_nodes()
    nodes.sort(key=ptree.node_sort)
    for node in nodes:
        t_ids.append(node.identifier)
        # add token edge
        dep_edges.append([node.identifier, node.identifier])  # self join
        # parent
        pnode = ptree.dp_tree.parent(node.identifier)
        if pnode and pnode.identifier != DROOT:
            dep_edges.append([node.identifier, pnode.identifier])
            dep_edges.append([pnode.identifier, node.identifier])
        # children
        child_nodes = ptree.dp_tree.children(node.identifier)
        for child_node in child_nodes:
            dep_edges.append([node.identifier, child_node.identifier])
            dep_edges.append([child_node.identifier, node.identifier])
    return t_ids,dep_edges

def remove_repetion(llist):
    new_list = []
    for li in llist:
        if li not in new_list:
            new_list.append(li)
    return new_list

def build_adj(sour_edges,t_ids):
    ids = np.array(t_ids, dtype=np.int32)
    matrix_shape = np.array(t_ids).shape[0]

    idx_map = {j: i for i, j in enumerate(ids)}
    edges = []
    for i,edge in enumerate(sour_edges):
        edges.append([idx_map[edge[0]], idx_map[edge[1]]])
    edges = np.array(edges, dtype=np.int32).reshape(np.array(sour_edges).shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(matrix_shape, matrix_shape),dtype=np.float32)

    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj

def load_and_cache_vocabs(examples, args):
    '''
    Build vocabulary of words and cache them.
    Load glove embedding if needed.
    '''
    embedding_cache_path = os.path.join(args.cache_dir, 'embedding')
    if not os.path.exists(embedding_cache_path):
        os.makedirs(embedding_cache_path)

    # Build or load word vocab and word2vec embeddings.
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join(
            embedding_cache_path, 'cached_{}_word_vocab.pkl'.format(args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',args.dataset_name)
            word_vocab = build_text_vocab(examples)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(embedding_cache_path, 'cached_{}_word_vecs.pkl'.format(args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.embedding_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.embedding_dir, 0.25, args.token_embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        cached_word_vocab_file = os.path.join(
            embedding_cache_path, 'cached_{}_word_vocab.pkl'.format(args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s', args.dataset_name)
            word_vocab = build_text_vocab(examples)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(embedding_cache_path,
                                             'cached_{}_word_vecs.pkl'.format(args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.embedding_dir)
            word_vecs = load_word2vec_embedding(word_vocab['itos'], args, 0.25)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)

    return word_vecs, word_vocab

def load_word2vec_embedding(words,args,uniform_scale):
    path = os.path.join(args.embedding_dir,'baike_26g_news_13g_novel_229g.model')
    w2v_model = gensim.models.Word2Vec.load(path)
    w2v_vocabs = [word for word, Vocab in w2v_model.wv.vocab.items()]  # 存储 所有的 词语

    word_vectors = []
    for word in words:
        if word in w2v_vocabs:
            word_vectors.append(w2v_model.wv[word])
        elif word == '[PAD]':
            word_vectors.append(np.zeros(w2v_model.vector_size, dtype=np.float32))
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, w2v_model.vector_size))

    return word_vectors

def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r',encoding='utf-8') as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                glove_dir, 'glove.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '[PAD]':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def _default_unk_index():
    return 1

def build_text_vocab(examples, vocab_size=1000000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['tokens'])

    itos = ['[PAD]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class ED_Dataset(Dataset):
    def __init__(self, examples,args,word_vocab):
        self.examples = examples
        self.args = args
        self.word_vocab = word_vocab

        self.convert_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        items = e['token_ids'],e['event_ids'],e['token_adj'],e['phrase_adj'],e['structure_adj'],\
                e['event_adj'],e['token2phrase'],e['phrase2structure'],e['structure2event'],e['label']

        return items
        # items_tensor = tuple(torch.tensor(t) for t in items)
        # return items_tensor

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.examples)):
            self.examples[i]['event_ids'] = []
            self.examples[i]['token_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['tokens']]
            for event in self.examples[i]['events']:
                event_token = [self.word_vocab['stoi'][w] for w in event]
                self.examples[i]['event_ids'].append(event_token)


def my_collate(batch):
    '''
    Pad event in a batch.
    Sort the events based on length.
    Turn all into tensors.
    '''
    # from Dataset.__getitem__()
    token_ids,event_ids,token_adj,phrase_adj,structure_adj,\
    event_adj,token2phrase,phrase2structure,structure2event,labels = zip(
        *batch)

    global token_max_len, phrase_max_len, structure_max_len, event_max_len
    # Pad sequences.
    token_ids = torch.tensor(pad_sequences(token_ids,maxlen=token_max_len,dtype=np.long,padding='post',truncating='post',value=0),dtype=torch.long)
    event_ids = padding_3dim(event_ids,token_max_len,event_max_len)

    token_adj = padding_user(token_adj,token_max_len,token_max_len)
    phrase_adj = padding_user(phrase_adj,phrase_max_len,phrase_max_len)
    structure_adj = padding_user(structure_adj,structure_max_len,structure_max_len)
    event_adj = padding_user(event_adj,event_max_len,event_max_len)

    token2phrase = padding_user(token2phrase,phrase_max_len,token_max_len)
    phrase2structure = padding_user(phrase2structure,structure_max_len,phrase_max_len)
    structure2event = padding_user(structure2event,1,structure_max_len)

    labels = torch.tensor(labels)

    return token_ids,event_ids,token_adj,phrase_adj,structure_adj,\
    event_adj,token2phrase,phrase2structure,structure2event,labels

# def padding_adj(sour_adj,token_ids):
#     adj_list = []
#     for i, t in enumerate(sour_adj):
#         pad = nn.ZeroPad2d(padding=(0, token_ids.shape[1] - t.shape[1], 0, token_ids.shape[1] - t.shape[1]))
#         adj_list.append(pad(t))
#     new_adj = torch.stack(adj_list, dim=0)
#
#     return new_adj

def padding_user(source,right,bottom):
    temp = []
    for events in source:
        events = torch.tensor(events)
        pad = nn.ZeroPad2d(padding=(0,right - events.shape[1],0,bottom - events.shape[0]))
        temp.append(pad(events))
    tgt = torch.stack(temp, dim=0)
    return tgt

def padding_3dim(source,token_max_len,event_max_len):
    batch_tensor_list = []
    for i,events in enumerate(source):
        events_tensor_list = []
        for j,event in enumerate(events):
            for i in range(token_max_len-len(event)):
                event.append(0)
            events_tensor_list.append(torch.tensor(event[0:token_max_len]))
        event_tensor = torch.stack(events_tensor_list,dim=0)
        pad = nn.ZeroPad2d(padding=(0, token_max_len-event_tensor.shape[1], 0, event_max_len - event_tensor.shape[0]))
        batch_tensor_list.append(pad(event_tensor))
    tgt = torch.stack(batch_tensor_list,dim=0)
    return tgt