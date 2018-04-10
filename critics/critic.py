from keras import layers, models, optimizers
from keras import backend as K

class Critic():
    def __init__(self, state_size, action_size, network_cfg):
        """

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param network_cfg: Dictionary of parameters to build the hidden layers of a neural network
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model(network_cfg)

    def build_model(self, network_cfg):
        # Placeholder
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden layers
        net_states = None
        for i, size in enumerate(network_cfg['hidden_size_states']):
            previous = net_states if i > 0 else states
            net_states = layers.Dense(units=size, activation='relu', name='dense_state_' + str(i))(previous)

        net_actions = None
        for i, size in enumerate(network_cfg['hidden_size_actions']):
            previous = net_actions if i > 0 else actions
            net_actions = layers.Dense(units=size, activation='relu', name='dense_action_' + str(i))(previous)

        # Combine pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Q_values output
        outputs = layers.Dense(units=1, name='q_values')(net)

        # Keras model
        self.model = models.Model(inputs=[states, actions], outputs=outputs)

        # Loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(outputs, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)