"""
The code is modified from morvanzhou's github and his tutorial page: https://morvanzhou.github.io/tutorials/
Notice: Although there are some codes related to priority experience replay, I forbid this function in main
program. In other words, it is completely random replay.
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
keep_pro = tf.placeholder(tf.float32)


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

lr = 0.00025
class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=lr,
            reward_decay=0.99,
            e_greedy=1.0,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
            epsilon=0.,
            memory_counter = 0,
            learn_step_counter=1,
            total_loss = 0.,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon #if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = learn_step_counter
        self.memory_counter = memory_counter
        self.total_loss = total_loss
        self.train = True
        self.keep_pro = 0.99

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def initiate_common_par(self):
        # self.n_actions = n_actions
        # self.n_features = n_features
        self.lr = lr
        # self.gamma = 0.99
        self.epsilon_max = 1.0
        # self.replace_target_iter = 500
        # self.memory_size = 10000
        # self.batch_size = 32
        self.epsilon_increment = 0.0001
        self.epsilon = 0.  # if e_greedy_increment is not None else self.epsilon_max

        # self.prioritized = prioritized  # decide to use double q or not

        self.learn_step_counter = 1
        self.memory_counter = 0
        self.total_loss = 0

    def decision(self, train_flag):
        if train_flag:
            self.keep_pro = 0.999
        else:
            self.keep_pro = 1.0
        return self.keep_pro

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, keep_pro):
            with tf.variable_scope('l1_eva'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                l1_dropout = tf.nn.dropout(l1, keep_prob=keep_pro)

            with tf.variable_scope('l2_eva'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
                l2_dropout = tf.nn.dropout(l2, keep_prob=keep_pro)

            with tf.variable_scope('l3_eva'):
                w3 = tf.get_variable('w3', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3
                out_dropout = tf.nn.dropout(out, keep_prob=keep_pro)
                # tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(0.01/32)(w3))
                # if trainable:
                #     tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0)(w2))
                #     tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(b2))
            return out_dropout

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        # self.q_evalu = tf.placeholder(tf.float32, [None, self.n_actions])
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 256, \
                tf.contrib.layers.xavier_initializer(), tf.constant_initializer(0.)  # config of layers
################tf.contrib.layers.xavier_initializer()tf.random_normal_initializer(0., 0.3)
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, keep_pro)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:

                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
                # tf.add_to_collection('losses', self.loss)
        with tf.variable_scope('train'):
            # self.total_loss = tf.add_n(tf.get_collection('losses'))
            # total_loss = re_loss+self.loss
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, keep_pro)

    def store_transition(self, s, a, r, s_):#-----------------------------------------
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation, initial):
        observation = observation[np.newaxis, :]
        if initial:
            action = np.random.randint(0, self.n_actions)
        else:
            if np.random.uniform() < self.epsilon:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation,
                                                                      keep_pro: self.keep_pro})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next = self.sess.run(
                self.q_next,
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           keep_pro: self.keep_pro})

        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={self.s: batch_memory[:, :self.n_features],
                       keep_pro: self.keep_pro})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        for _ in range(0, 32):
            q_target[batch_index[_], eval_act_index[_]] = reward[_] + self.gamma * np.max(q_next[_, :]) if reward[_] ==-1.0 else -1.0
        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) #if reward[_] ==1 else 1.0

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    keep_pro: self.keep_pro})

        self.cost_his.append(self.loss)

        if self.learn_step_counter % 1000 == 0:
            print(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        if self.learn_step_counter%1000==0:
            print(self.learn_step_counter)
        self.learn_step_counter += 1
