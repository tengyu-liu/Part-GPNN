import tensorflow as tf

tf.flags.DEFINE_string('name', 'exp', '')
tf.flags.DEFINE_integer('restore_epoch', -1, '')

tf.flags.DEFINE_integer('epochs', 100, '')
tf.flags.DEFINE_integer('batch_size', 1, '')

tf.flags.DEFINE_integer('node_num', 517, '')    # 517 for global max, 433 for train max
tf.flags.DEFINE_integer('node_feature_size', 1000, '')
tf.flags.DEFINE_integer('edge_feature_size', 1024, '')
tf.flags.DEFINE_integer('label_num', 30, '')
tf.flags.DEFINE_integer('message_passing_iterations', 3, '')

tf.flags.DEFINE_bool('debug', False, '')

tf.flags.DEFINE_float('lr', 1e-3, '')
tf.flags.DEFINE_float('beta1', 0.99, '')
tf.flags.DEFINE_float('beta2', 0.999, '')

tf.flags.DEFINE_integer('log_interval', 10, '')

flags = tf.flags.FLAGS
