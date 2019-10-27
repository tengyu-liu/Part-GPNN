import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, config):
        self.load_config(config)
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()

    def load_config(self, config):
        self.node_num = config.node_num
        self.node_feature_size = config.node_feature_size
        self.edge_feature_size = config.edge_feature_size
        self.label_num = config.label_num
        self.message_passing_iterations = config.message_passing_iterations 
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        pass

    def build_input(self):
        self.batch_node_num = tf.placeholder(tf.int32, [], 'batch_node_num')
        self.node_features = tf.placeholder(tf.float32, [None, self.node_num, self.node_feature_size], 'node_features')
        # We need an edge feature as the initial value for message passing
        self.edge_features = tf.placeholder(tf.float32, [None, self.node_num, self.node_num, self.edge_feature_size], 'edge_features') 
        self.adj_mat = tf.placeholder(tf.float32, [None, self.node_num, self.node_num], 'adj_mat')
        self.pairwise_label_gt = tf.placeholder(tf.float32, [None, self.node_num, self.node_num, self.label_num], 'pairwise_label_gt')
        self.gt_strength_level = tf.placeholder(tf.float32, [None, self.node_num, self.node_num], 'gt_strength_level')

        p = tf.print('Batch node num:', self.batch_node_num)
        with tf.control_dependencies([p]):
            self._node_features = self.node_features[:,:self.batch_node_num, :]
            self._edge_features = self.edge_features[:,:self.batch_node_num, self.batch_node_num, :]
            self._adj_mat = self.adj_mat[:,:self.batch_node_num, self.batch_node_num]
            self._pairwise_label_gt = self.pairwise_label_gt[:,:self.batch_node_num, self.batch_node_num,:]
            self._gt_strength_level = self.gt_strength_level[:,:self.batch_node_num, self.batch_node_num]

    def build_model(self):

        self.step = tf.Variable(0)

        self.update_module = tf.keras.layers.GRUCell(self.node_feature_size)
        
        for message_passing_iteration in range(self.message_passing_iterations):
            if message_passing_iteration == 0:
                message = self.message(self._node_features, self._edge_features, self._adj_mat)
                hidden_node_feature = self.update(message, self._node_features)
            else:
                message = self.message(hidden_node_feature, message, self._adj_mat)
                hidden_node_feature = self.update(message, hidden_node_feature)

        self.edge_label_pred = self.readout(message)

    def build_train(self):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._pairwise_label_gt, logits=self.edge_label_pred) * tf.expand_dims(self._gt_strength_level, axis=-1), axis=[1,2,3]) / tf.reduce_sum(self._gt_strength_level, axis=[1,2]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.step)
    
    def build_summary(self):
        self.summ_loss_in = tf.placeholder(tf.float32, [], 'loss_summ')
        self.summ_map_sum_in = tf.placeholder(tf.float32, [], 'mAP_sum_summ')
        self.summ_map_max_in = tf.placeholder(tf.float32, [], 'mAP_max_summ')
        self.summ_map_mean_in = tf.placeholder(tf.float32, [], 'mAP_mean_summ')
        _ = tf.summary.scalar('loss', self.summ_loss_in)
        _ = tf.summary.scalar('mAP_sum', self.summ_map_sum_in)
        _ = tf.summary.scalar('mAP_max', self.summ_map_max_in)
        _ = tf.summary.scalar('mAP_mean', self.summ_map_mean_in)
        self.summ = tf.summary.merge_all()
    
    def message(self, node_features, edge_features, adjacency_matrix):
        """
        Computes messages according to current node features and edge features, as well as adjacency matrices
        M_ij = A_ij L(N_i, N_j, M_ij)

        Arguments:
        node_features: B x N x F_n
        edge_features: B x N x N x F_m
        adjacency_matrix: B x N x N x 1
        Returns:
        message: B x N x N x F_m 
        """
        node_message = tf.layers.dense(node_features, self.edge_feature_size / 4, activation=tf.nn.relu)
        edge_message = tf.layers.dense(edge_features, self.edge_feature_size / 2, activation=tf.nn.relu)
        node_message_left = tf.tile(tf.expand_dims(node_message, axis=2), [1,1,self.node_num,1])
        node_message_right = tf.tile(tf.expand_dims(node_message, axis=1), [1,self.node_num,1,1])
        print(node_message_left, node_message_right, edge_message)
        message = tf.concat([node_message_left, node_message_right, edge_message], axis=-1)
        message = message * tf.expand_dims(adjacency_matrix, axis=-1)
        return message
    
    def update(self, messages, node_features):
        """
        Updates node features according to messages
        N_i = F(\sum_j M_ij, N_i)
            where F is a GRU module
        
        Arguments:
        messages: B x N x N x F_m
        node_features: B x N x F_n
        Returns:
        node_features: B x N x F_n
        """
        messages = tf.reshape(tf.reduce_sum(messages, axis=2), [-1, self.edge_feature_size])                            # BN x Fm
        node_features = tf.reshape(node_features, [-1, self.node_feature_size])                                         # BN x Fn
        _, node_features = self.update_module(messages, [node_features], training=True)
        node_features = tf.reshape(node_features, [-1, self.node_num, self.node_feature_size])
        return node_features    

    def readout(self, edge_features):
        hidden = tf.layers.dense(edge_features, self.edge_feature_size, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, self.label_num, activation=None)
        return output

if __name__ == "__main__":
    from config import flags as config
    model = Model(config)