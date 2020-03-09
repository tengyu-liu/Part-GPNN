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
        self.node_feature_size = config.node_feature_size
        self.edge_feature_size = config.edge_feature_size
        self.label_num = config.label_num
        self.message_passing_iterations = config.message_passing_iterations 
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.dropout = config.dropout
        self.dataset = config.dataset
        pass

    def build_input(self):
        self.training = tf.placeholder(tf.bool, [], 'training')
        self.batch_node_num = tf.placeholder(tf.int32, [], 'batch_node_num')
        self.node_features = tf.placeholder(tf.float32, [None, None, self.node_feature_size], 'node_features')
        # We need an edge feature as the initial value for message passing
        self.edge_features = tf.placeholder(tf.float32, [None, None, None, self.edge_feature_size], 'edge_features') 
        self.adj_mat = tf.placeholder(tf.float32, [None, None, None], 'adj_mat')
        self.pairwise_label_gt = tf.placeholder(tf.float32, [None, None, None, self.label_num], 'pairwise_label_gt')
        self.gt_strength_level = tf.placeholder(tf.float32, [None, None, None], 'gt_strength_level')
        self.pairwise_label_mask = tf.placeholder(tf.float32, [None, None, None, self.label_num], 'pairwise_label_mask')
        if self.dataset == 'vcoco':
            self.pairwise_role_gt = tf.placeholder(tf.float32, [None, None, None, self.label_num, 3], 'pairwise_role_gt')

    def build_model(self):

        self.step = tf.Variable(0)

        self.message_module = [tf.layers.Dense(self.edge_feature_size / 4, activation=tf.nn.relu), tf.layers.Dense(self.edge_feature_size / 2, activation=tf.nn.relu)]
        self.update_module = tf.keras.layers.GRUCell(self.node_feature_size, dropout=self.dropout)
        
        for message_passing_iteration in range(self.message_passing_iterations):
            if message_passing_iteration == 0:
                message = self.message(self.node_features, self.edge_features, self.adj_mat)
                hidden_node_feature = self.update(message, self.node_features)
            else:
                message = self.message(hidden_node_feature, message, self.adj_mat)
                hidden_node_feature = self.update(message, hidden_node_feature)

        self.edge_label = self.readout(message, self.label_num)
        self.edge_label_pred = tf.sigmoid(self.edge_label) * self.pairwise_label_mask

        if self.dataset == 'vcoco':
            edge_role = self.readout(message, 3)
            self.edge_role = tf.expand_dims(edge_role, axis=-2) + tf.expand_dims(self.edge_label, axis=-1)
            self.edge_role_pred = tf.sigmoid(self.edge_role)

    def build_train(self):
        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.pairwise_label_gt, 
            logits=self.edge_label, 
            weights=self.pairwise_label_mask * tf.expand_dims(self.gt_strength_level, axis=-1)) / tf.reduce_sum(self.gt_strength_level)
        if self.dataset == 'vcoco':
            role_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.pairwise_role_gt, 
                logits=self.edge_role, 
                weights=self.pairwise_label_mask * tf.expand_dims(self.gt_strength_level, axis=-1)) / tf.reduce_sum(self.gt_strength_level)
        p1 = tf.print('\n', loss, role_loss)
        p2 = tf.print('\n', tf.reduce_mean(loss), tf.reduce_mean(role_loss))
        with tf.control_dependencies([p1, p2]):
            self.loss = tf.reduce_mean(loss + role_loss)
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

        if self.dataset == 'hico':
            self.summ_part_map_sum_in = tf.placeholder(tf.float32, [], 'part_mAP_sum_summ')
            self.summ_part_map_max_in = tf.placeholder(tf.float32, [], 'part_mAP_max_summ')
            self.summ_part_map_mean_in = tf.placeholder(tf.float32, [], 'part_mAP_mean_summ')
            _ = tf.summary.scalar('part_mAP_sum', self.summ_part_map_sum_in)
            _ = tf.summary.scalar('part_mAP_max', self.summ_part_map_max_in)
            _ = tf.summary.scalar('part_mAP_mean', self.summ_part_map_mean_in)

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
        node_message = self.message_module[0](node_features)
        edge_message = self.message_module[1](edge_features)
        node_message_left = tf.tile(tf.expand_dims(node_message, axis=2), [1,1,self.batch_node_num,1])
        node_message_right = tf.tile(tf.expand_dims(node_message, axis=1), [1,self.batch_node_num,1,1])
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
        _, node_features = self.update_module(messages, [node_features], training=self.training)
        node_features = tf.reshape(node_features, [-1, self.batch_node_num, self.node_feature_size])
        return node_features    

    def readout(self, edge_features, output_num):
        hidden = tf.layers.dense(edge_features, self.edge_feature_size, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, output_num, activation=None)
        return output

if __name__ == "__main__":
    from config import flags as config
    model = Model(config)