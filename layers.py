import tensorflow as tf

from . import inits

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            x = dropout_sparse(inputs[j], 1-self.dropout, self.nonzero_feat[self.edge_type[1]])
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, axis=1)
        return outputs


class GraphConvolutionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mats, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            x = tf.nn.dropout(inputs[j], 1-self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)

            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, axis=1)
        return outputs

class GraphConvolutionMulti2(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mats, lat_adj_mats, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti2, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.lat_adj_mats = lat_adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            # self.vars['weights'] = inits.weight_variable_glorot(
            #     input_dim, output_dim, name='weights')
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k)
                # self.vars['a_%d' % k] = inits.weight_variable_glorot(
                #     1, 1, name='a_%d' % k)
                self.vars['a_%d' % k] = tf.get_variable(name='a_%d' % k, shape=[1], initializer=tf.ones_initializer())

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        a = []
        for k in range(self.num_types):
            x = tf.nn.dropout(inputs[j], 1-self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' % k])

            if not FLAGS.train_latent:
                f1 = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
                x = f1
            else:
                adj_mats = self.adj_mats[self.edge_type][k] * 1.0
                f1 = tf.sparse_tensor_dense_matmul(adj_mats, x)

                a_k = self.vars['a_%d' % k]
                a_k = tf.nn.sigmoid(a_k)
                # a_k = tf.constant(1.0)
                a.append(a_k)
                lat_adj_mats = self.lat_adj_mats[self.edge_type][k] * a_k
                f2 = tf.sparse_tensor_dense_matmul(lat_adj_mats, x)

                x = f1 + f2
                # x = tf.concat([f1, f2], 1)



            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, axis=1)
        return outputs, a


class GraphAttentionLayer(MultiLayer):
    def __init__(self, bias_mat, num_feat, hid_units, activation,
                 dropout=0.0, residual=False, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.bias_mat = bias_mat
        self.num_feat = num_feat
        self.hid_units = hid_units
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights_i'] = inits.weight_variable_glorot(
                self.num_feat[self.edge_type[0]], self.hid_units, name='weights_i')
            self.vars['weights_j'] = inits.weight_variable_glorot(
                self.num_feat[self.edge_type[1]], self.hid_units, name='weights_j')
            self.vars['a_i'] = inits.weight_variable_glorot(
                self.hid_units, 1, name='a_i')
            self.vars['a_j'] = inits.weight_variable_glorot(
                self.hid_units, 1, name='a_j')

    def _call(self, inputs=()):
        h_i = inputs[0]
        h_j = inputs[1]

        if self.dropout != 0.0:
            h_i = tf.nn.dropout(h_i, 1.0 - self.dropout)
            h_j = tf.nn.dropout(h_j, 1.0 - self.dropout)
        wh_i = tf.matmul(h_i, self.vars['weights_i'])
        wh_j = tf.matmul(h_j, self.vars['weights_j'])

        awh_i = tf.matmul(wh_i, self.vars['a_i'])
        awh_j = tf.matmul(wh_j, self.vars['a_j'])
        logits = awh_i + tf.transpose(awh_j, [1, 0])
        coefs = tf.nn.softmax(tf.sparse_add(tf.nn.leaky_relu(logits), self.bias_mat))

        # if self.dropout != 0.0:
        #     coefs = tf.nn.dropout(coefs, 1.0 - self.dropout)
        #     wh_j = tf.nn.dropout(wh_j, 1.0 - self.dropout)

        z_i = tf.matmul(coefs, wh_j)
        z_i = tf.contrib.layers.bias_add(tf.reshape(z_i, [-1, self.hid_units]))

        # residual connection
        if self.residual:
            if h_i.shape[-1] != z_i.shape[-1]:
                z_i = z_i + tf.layers.conv1d(h_i, z_i.shape[-1], 1)
            else:
                z_i = z_i + h_i

        return self.activation(z_i)



class DEDICOMDecoder(MultiLayer):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DEDICOMDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['global_interaction'] = inits.weight_variable_glorot(
                input_dim, input_dim, name='global_interaction')
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='local_variation_%d' % k)
                self.vars['local_variation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            relation = tf.diag(self.vars['local_variation_%d' % k])
            product1 = tf.matmul(inputs_row, relation)
            product2 = tf.matmul(product1, self.vars['global_interaction'])
            product3 = tf.matmul(product2, relation)
            rec = tf.matmul(product3, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='relation_%d' % k)
                self.vars['relation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            relation = tf.diag(self.vars['relation_%d' % k])
            intermediate_product = tf.matmul(inputs_row, relation)
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['relation_%d' % k] = inits.weight_variable_glorot(
                    input_dim, input_dim, name='relation_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            intermediate_product = tf.matmul(inputs_row, self.vars['relation_%d' % k])
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            rec = tf.matmul(inputs_row, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs
