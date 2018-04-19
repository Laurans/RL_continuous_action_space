from tensorflow import keras

class Critic():
    def __init__(self, state_shape, action_shape, network_cfg):
        """

        :param state_shape: Shape of each state
        :param action_shape: Dimension of each action
        :param network_cfg: Dictionary of parameters to build the hidden layers of a neural network
        """
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.build_model(network_cfg)

    def build_model(self, network_cfg):

        # Inputs
        states_inputs = keras.Input(shape=self.state_shape, name='states')
        actions_inputs = keras.Input(shape=self.action_shape,  name='actions')

        # State Branch
        rnn_layer = keras.layers.SimpleRNN(units=network_cfg['rnn_units'], return_sequences=False)(states_inputs)

        # Action Branch
        batch_norm = keras.layers.BatchNormalization()(actions_inputs)

        # Union
        concat = keras.layers.concatenate([batch_norm, rnn_layer])

        net = keras.layers.Dense(32, activation='relu')(concat)
        # Final Branch and output
        q_values = keras.layers.Dense(1)(net)

        # Keras model
        self.model = keras.models.Model(inputs=[states_inputs, actions_inputs], outputs=q_values)

        # Optimizer
        optimizer = keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Definition of action gradient
        action_gradients = keras.backend.gradients(q_values, actions_inputs)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = keras.backend.function(
            inputs=[*self.model.input, keras.backend.learning_phase()],
            outputs=action_gradients)
