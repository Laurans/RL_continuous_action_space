from utils.function import Function
import tensorflow as tf

class Actor():

    def __init__(self, state_shape, action_shape, action_range, tau, batch_size, network_cfg):
        """ Initialize parameters and build model
        """

        self.state_shape = [None] + state_shape
        self.action_shape = [None] + action_shape
        self.batch_size = batch_size
        self.tau = tf.constant(tau)

        action_low, action_high = action_range
        network_cfg['action_range'] = tf.constant(action_high - action_low)
        network_cfg['action_low'] = tf.constant(action_low)

        # Initialize a model
        self.build_model(network_cfg)

    def build_model(self, network_cfg):
        with tf.variable_scope('actor'):
            actor_local = ActorNetwork(
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                name='local',
                network_cfg=network_cfg,
                reuse=True
            )

            actor_local_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=tf.get_variable_scope().name + '/local')

            actor_target = ActorNetwork(
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                name='target',
                network_cfg=network_cfg,
            )
            actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope=tf.get_variable_scope().name + '/target')

            ## Create a copy function
            copy_var_local_into_target = []
            for var, var_target in zip(sorted(actor_local_vars, key=lambda v: v.name),
                                       sorted(actor_target_vars, key=lambda v: v.name)):
                copy_var_local_into_target.append(var_target.assign(var))

            copy_var_local_into_target = tf.group(*copy_var_local_into_target)

            self.copy_local_in_target = Function(updates=[copy_var_local_into_target])

            ## Create a soft update function
            update_var_target = []
            for var, var_target in zip(sorted(actor_local_vars, key=lambda v: v.name),
                                       sorted(actor_target_vars, key=lambda v: v.name)):
                update_var_target.append(var_target.assign(self.tau * var + (1 - self.tau) * var_target))

            self.soft_update = Function(updates=[update_var_target])

            ## Create a predict function
            self.predict = Function(inputs=[actor_local.states_inputs],
                                    outputs=[actor_local.output])

            ## Create a function to get target
            self.get_targets = Function(inputs=[actor_target.states_inputs],
                                    outputs=[actor_target.output])

            ## Create a train function

            trainable_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/local')
            actor_grad = tf.gradients(actor_local.output, trainable_vars, grad_ys=-actor_local.actions_grad)
            actor_grad = list(map(lambda x: tf.div(x, self.batch_size), actor_grad))

            optimizer = actor_local.train_op.apply_gradients(zip(actor_grad, trainable_vars))

            self.fit = Function(inputs=[actor_local.states_inputs, actor_local.actions_grad],
                               updates=[optimizer])


class ActorNetwork:
    def __init__(self, state_shape, action_shape, name, network_cfg, reuse=False):
        with tf.variable_scope(name) as scope:
            # Inputs
            self.states_inputs = tf.placeholder(tf.float32, state_shape, name='states_inputs')

            # Target
            self.actions_grad = tf.placeholder(tf.float32, action_shape, name='actions_grad_inputs')

            # Network
            net = self.states_inputs

            for hidden_units in network_cfg['layers']:
                net = tf.layers.batch_normalization(net)
                net = tf.layers.dense(net, hidden_units, activation=tf.nn.relu)

            # Output
            raw_actions = tf.layers.dense(net, action_shape[-1], activation=tf.nn.sigmoid, name='raw_actions')
            self.output = network_cfg['action_range'] * raw_actions + network_cfg['action_low']

            # Optimizer
            self.train_op = tf.train.AdamOptimizer(network_cfg['learning_rate'])

            if reuse:
                scope.reuse_variables()



