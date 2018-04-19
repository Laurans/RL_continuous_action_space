import tensorflow as tf

class Function():
    def __init__(self, inputs=[], outputs=[], updates=[]):
        self.inputs = inputs
        self.update_group = tf.group(*updates)
        self.outputs_update = outputs + [self.update_group]

    def __call__(self, *args):
        feed_dict = {}

        for inpt, value in zip(self.inputs, args):
            feed_dict[inpt] = value

        results = tf.get_default_session().run(self.outputs_update,
                                               feed_dict=feed_dict)[0]
        return results
