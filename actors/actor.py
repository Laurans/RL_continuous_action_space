from tensorflow import keras
import tensorflow as tf

class Actor():

    def __init__(self, state_shape, action_shape, action_low, action_high, network_cfg):
        """ Initialize parameters and build model

        :param state_shape: Shape of each state
        :param action_shape: Dimension of each action
        :param action_low: Min value of each action
        :param action_high: Max value of each action
        :param network_cfg: Dictionary of parameters to build the hidden layers of a neural network
        """

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        # Initialize a model
        self.build_model(network_cfg)

    def build_model(self, network_cfg):
        
        #  Inputs
        states_inputs = keras.Input(shape=self.state_shape)
        actions_gradients_inputs = keras.Input(shape=self.action_shape)

        # Neural network core
        rnn_layer2 = keras.layers.SimpleRNN(units=network_cfg['rnn_units'], return_sequences=False)(states_inputs)

        # Outpus
        raw_actions = keras.layers.Dense(1, activation='sigmoid', name='raw_actions')(rnn_layer2)
        outputs = keras.layers.Lambda(lambda x: x * self.action_range + self.action_low, name='output_actions')(raw_actions)

        # Keras model
        self.model = keras.models.Model(inputs=states_inputs, outputs=outputs)


        # Loss function
        loss = keras.backend.mean(-actions_gradients_inputs * outputs)

        # Define optimizer and training function
        optimizer = keras.optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras.backend.function(
            inputs=[self.model.input, actions_gradients_inputs, keras.backend.learning_phase()],
            outputs=[loss],
            updates=updates_op
        )