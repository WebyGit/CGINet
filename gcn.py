from collections import defaultdict

import tensorflow as tf

from .layers import GraphConvolutionSparseMulti, GraphConvolutionMulti, GraphConvolutionMulti2, \
    DistMultDecoder, InnerProductDecoder, DEDICOMDecoder, BilinearDecoder

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModel(Model):
    def __init__(self, edge_types1, edge_types2, decoders, num_feat, nonzero_feat, placeholders, **kwargs):
        super(GCNModel, self).__init__(**kwargs)
        self.edge_types1 = edge_types1
        self.edge_types2 = edge_types2
        self.decoders = decoders
        self.placeholders = placeholders
        self.num_feat = num_feat
        self.nonzero_feat = nonzero_feat
        self.num_obj_types1 = max([i for i, _ in self.edge_types1]) + 1
        self.num_obj_types2 = max([i for i, _ in self.edge_types2]) + 1
        self.dropout = placeholders['dropout']
        self.a = {}

        self.inputs1 = {i: placeholders['feat_%d' % i] for i, _ in self.edge_types1}
        self.adj_mats1 = {et: [
            placeholders['adj_mats1_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
            for et, n in self.edge_types1.items()}
        self.adj_mats2 = {et: [
            placeholders['adj_mats2_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
            for et, n in self.edge_types2.items()}
        if FLAGS.train_latent:
            self.lat_adj_mats2 = {et: [
                placeholders['lat_adj_mats2_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
                for et, n in self.edge_types2.items()}
        else:
            self.lat_adj_mats2 = {}
        self.build()

    def _build(self):

        # chem-chem; chem-pathway; gene-gene; gene-pathway; pathway-chem; pathway-gene
        with tf.name_scope('%s_external_embed' % self.name):
            with tf.name_scope('layer_1'):
                self.hidden1_1 = defaultdict(list)
                for i, j in self.edge_types1:
                    self.hidden1_1[i].append(GraphConvolutionSparseMulti(
                        input_dim=self.num_feat[j],
                        output_dim=FLAGS.hid_units1_1,
                        edge_type=(i, j),
                        num_types=self.edge_types1[i, j],
                        adj_mats=self.adj_mats1,
                        nonzero_feat=self.nonzero_feat,
                        dropout=self.dropout,
                        act=lambda x: x,
                        logging=self.logging)(self.inputs1))
                for i, hid in self.hidden1_1.items():
                    self.hidden1_1[i] = tf.nn.relu(tf.add_n(hid))

            with tf.name_scope('layer_2'):
                self.hidden1_2 = defaultdict(list)
                for i, j in self.edge_types1:
                    self.hidden1_2[i].append(GraphConvolutionMulti(
                        input_dim=FLAGS.hid_units1_1,
                        output_dim=FLAGS.hid_units1_2,
                        edge_type=(i, j),
                        num_types=self.edge_types1[i, j],
                        adj_mats=self.adj_mats1,
                        dropout=self.dropout,
                        act=lambda x: x,
                        logging=self.logging)(self.hidden1_1))
                for i, hid in self.hidden1_2.items():
                    self.hidden1_2[i] = tf.nn.relu(tf.add_n(hid))

        # chem-gene; gene-chem
        with tf.name_scope('%s_inner_embed' % self.name):
            with tf.name_scope('layer_1'):
                self.hidden2_1 = defaultdict(list)
                for i, j in self.edge_types2:
                    outs, b = GraphConvolutionMulti2(
                        input_dim=FLAGS.hid_units1_2,
                        output_dim=FLAGS.hid_units2_1,
                        edge_type=(i, j),
                        num_types=self.edge_types2[i, j],
                        adj_mats=self.adj_mats2,
                        lat_adj_mats=self.lat_adj_mats2,
                        dropout=self.dropout,
                        act=lambda x: x,
                        logging=self.logging)(self.hidden1_2)
                    self.hidden2_1[i].append(outs)
                for i, hid in self.hidden2_1.items():
                    self.hidden2_1[i] = tf.nn.relu(tf.add_n(hid))
            with tf.name_scope('layer_2'):
                self.hidden2_2 = defaultdict(list)
                for i, j in self.edge_types2:
                    outputs, a = GraphConvolutionMulti2(
                        input_dim=FLAGS.hid_units2_1,
                        output_dim=FLAGS.hid_units2_2,
                        edge_type=(i, j),
                        num_types=self.edge_types2[i, j],
                        adj_mats=self.adj_mats2,
                        lat_adj_mats=self.lat_adj_mats2,
                        dropout=self.dropout,
                        act=lambda x: x,
                        logging=self.logging)(self.hidden2_1)
                    self.hidden2_2[i].append(outputs)
                    self.a[(i, j)] = a
                self.embeddings = [None] * self.num_obj_types2
                for i, hid in self.hidden2_2.items():
                    self.embeddings[i] = tf.add_n(hid)

        # decoder
        hid_units = FLAGS.hid_units2_2
        self.edge_type2decoder = {}
        for i, j in self.edge_types2:
            decoder = self.decoders[i, j]
            if decoder == 'innerproduct':
                self.edge_type2decoder[i, j] = InnerProductDecoder(
                    input_dim=hid_units, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types2[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'distmult':
                self.edge_type2decoder[i, j] = DistMultDecoder(
                    input_dim=hid_units, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types2[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'bilinear':
                self.edge_type2decoder[i, j] = BilinearDecoder(
                    input_dim=hid_units, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types2[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'dedicom':
                self.edge_type2decoder[i, j] = DEDICOMDecoder(
                    input_dim=hid_units, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types2[i, j],
                    act=lambda x: x, dropout=self.dropout)
            else:
                raise ValueError('Unknown decoder type')
        self.latent_inters = []
        self.latent_varies = []
        for edge_type in self.edge_types2:
            decoder = self.decoders[edge_type]
            for k in range(self.edge_types2[edge_type]):
                if decoder == 'innerproduct':
                    glb = tf.eye(hid_units, hid_units)
                    loc = tf.eye(hid_units, hid_units)
                elif decoder == 'distmult':
                    glb = tf.diag(self.edge_type2decoder[edge_type].vars['relation_%d' % k])
                    loc = tf.eye(hid_units, hid_units)
                elif decoder == 'bilinear':
                    glb = self.edge_type2decoder[edge_type].vars['relation_%d' % k]
                    loc = tf.eye(hid_units, hid_units)
                elif decoder == 'dedicom':
                    glb = self.edge_type2decoder[edge_type].vars['global_interaction']
                    loc = tf.diag(self.edge_type2decoder[edge_type].vars['local_variation_%d' % k])
                else:
                    raise ValueError('Unknown decoder type')
                self.latent_inters.append(glb)
                self.latent_varies.append(loc)
