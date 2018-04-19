import tensorflow as tf
from utils.function import Function


class Critic():
    def __init__(self, state_shape, action_shape, tau, network_cfg):
        """

        :param state_shape: Shape of each state
        :param action_shape: Dimension of each action
        :param network_cfg: Dictionary of parameters to build the hidden layers of a neural network
        """
        self.state_shape = [None] + state_shape
        self.action_shape = [None] + action_shape
        self.tau = tf.constant(tau)

        self.build_graph(network_cfg)

    def build_graph(self, network_cfg):
        with tf.variable_scope('critic'):
            critic_local = CriticNetwork(
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                name='local',
                network_cfg=network_cfg,
                reuse=True
            )
            critic_local_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=tf.get_variable_scope().name + '/local')

            critic_target = CriticNetwork(
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                name='target',
                network_cfg=network_cfg,
            )
            critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope=tf.get_variable_scope().name + '/target')

            ## Create a copy function
            copy_var_local_into_target = []
            for var, var_target in zip(sorted(critic_local_vars, key=lambda v: v.name),
                                       sorted(critic_target_vars, key=lambda v: v.name)):
                copy_var_local_into_target.append(var_target.assign(var))

            copy_var_local_into_target = tf.group(*copy_var_local_into_target)

            self.copy_local_in_target = Function(updates=[copy_var_local_into_target])

            ## Create a soft update function
            update_var_target = []
            for var, var_target in zip(sorted(critic_local_vars, key=lambda v: v.name),
                                       sorted(critic_target_vars, key=lambda v: v.name)):
                update_var_target.append(var_target.assign(self.tau * var + (1 - self.tau) * var_target))

            self.soft_update = Function(updates=[update_var_target])

            ## Create a predict function
            self.predict = Function(inputs=[critic_local.states_inputs, critic_local.actions_inputs],
                                    outputs=[critic_local.output])

            ## Create function to get target
            self.get_targets = Function(inputs=[critic_target.states_inputs, critic_target.actions_inputs],
                                    outputs=[critic_target.output])

            ## Create a train function
            self.fit = Function(inputs=[critic_local.states_inputs, critic_local.actions_inputs,
                                        critic_local.targets],
                                outputs=[critic_local.loss],
                                updates=[critic_local.optimizer])

            ## Create a getter to actions gradients
            self.get_actions_grad = Function(inputs=[critic_local.states_inputs, critic_local.actions_inputs],
                                             outputs=[critic_local.actions_gradients])


class CriticNetwork:
    def __init__(self, state_shape, action_shape, name, network_cfg, reuse=False, learning_rate=0.01):
        with tf.variable_scope(name) as scope:
            # Inputs
            self.states_inputs = tf.placeholder(tf.float32, state_shape, name='states_inputs')
            self.actions_inputs = tf.placeholder(tf.float32, action_shape, name='actions_inputs')

            # Target q values for training
            self.targets = tf.placeholder(tf.float32, [None, 1], name='target_outputs')

            # State Branch
            state_branch = tf.layers.dense(self.states_inputs, network_cfg['layers'][0],
                                           activation=tf.nn.relu)

            # Action branch
            action_branch = tf.layers.dense(self.actions_inputs, network_cfg['layers'][0],
                                            activation=tf.nn.relu)

            # Merge
            net = tf.add(state_branch, action_branch)

            # End of network
            for hidden_units in network_cfg['layers'][1:]:
                net = tf.layers.dense(net, hidden_units, activation=tf.nn.relu)

            # Output
            self.output = tf.layers.dense(net, 1, activation=None, name='output')

            # Loss
            self.loss = tf.losses.mean_squared_error(labels=self.targets,
                                                     predictions=self.output)
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.actions_gradients = tf.gradients(self.output, self.actions_inputs)

            if reuse:
                scope.reuse_variables()
