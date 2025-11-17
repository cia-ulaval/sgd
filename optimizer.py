from sympy import symbols, diff
import tensorflow as tf
class Optimizer:
    def __init__(self, weigths):
        self.weights = weigths
    def calculate_update(self, hidden, y_pred, y_true):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            )
        grads = tape.gradient(loss, [y_pred])








