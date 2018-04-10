from keras import layers, models, optimizers
from keras import backend as K

class Actor():

    def __init__(self, state_size, action_size, action_low, action_high, network_cfg):
        """ Initialize parameters and build model

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param action_low: Min value of each action
        :param action_high: Max value of each action
        :param network_cfg: Dictionary of parameters to build the hidden layers of a neural network
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        # Initialize a model
        self.build_model(network_cfg)

    def build_model(self, network_cfg):
        # Placeholder
        states = layers.Input(shape=(self.state_size,), name='states')

        # Hidden layers
        net = None
        for i, size in enumerate(network_cfg['hidden_size']):
            previous = net if i > 0 else states
            net = layers.Dense(units=size, activation='relu', name='dense_'+str(i))(previous)

        # Final output layer
        raw_actions = layers.Dense(units=self.action_size,
                                   activation='sigmoid', name='raw_actions')(net)

        # Output
        outputs = layers.Lambda(lambda x: x * self.action_range + self.action_low,
                                    name='actions')(raw_actions)

        # Keras model
        self.model = models.Model(inputs=states, outputs=outputs)


        # Loss function
        actions_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-actions_gradients * outputs)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, actions_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op
        )