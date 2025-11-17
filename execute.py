import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import model

import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
print(tf.__version__)
print(tf_build_info.build_info)
print("GPUs:", tf.config.list_physical_devices('GPU'))

HIDDEN_LAYER_SIZE = [128,64,32,16]
BATCH_SIZE = 32
EPOCHS = 45
learning_rate = 0.01
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)

train_X = train_X.reshape(train_X.shape[0], -1).astype('float32') / 255.0
val_X = val_X.reshape(val_X.shape[0], -1).astype('float32') / 255.0
test_X = test_X.reshape(test_X.shape[0], -1).astype('float32') / 255.0

modele = model.Model(784, HIDDEN_LAYER_SIZE, 10, learning_rate)

with open("training_results.txt", "a") as f:
    for noise in [True, 'True and proportionnal to weights', False]:
        for epoch in range(EPOCHS):
            for step in range(0, len(train_X), BATCH_SIZE):
                x_batch = tf.convert_to_tensor(train_X[step:step + BATCH_SIZE], dtype=tf.float32)
                y_batch = tf.convert_to_tensor(train_y[step:step + BATCH_SIZE], dtype=tf.int32)

                if noise is False:
                    loss= modele.normalSGD_train_step(x_batch, y_batch)
                elif noise == 'True and proportionnal to weights':
                    loss, samples = modele.Laplacian_train_step_with_prop_noise(x_batch, y_batch)
                else :
                    loss, samples = modele.Laplacian_train_step(x_batch, y_batch)

            _, _, _, val_pred = modele.forward(val_X)
            val_classes = tf.argmax(val_pred, axis=1, output_type=tf.int32)
            val_acc = tf.reduce_mean(tf.cast(val_classes == val_y, tf.float32))

            print(f"Epoch {epoch + 1}/{EPOCHS} completed.")
        print(
            f"Final Loss {'with Laplacian noise ' if noise else ''}= {loss.numpy():.4f}, {f"Number of samples = {samples}" if noise else ""}, Validation accuracy = {val_acc.numpy():.4f}")
        f.write(
            f"Final Loss {'with Laplacian noise ' if noise else ''}= {loss.numpy():.4f}, Proportional noise = {"True" if noise == 'True and proportionnal to weights' else "False"}"
            f"Hidden layers = {HIDDEN_LAYER_SIZE},  {f"Number of samples = {samples}, " if noise else ""} Number of Epochs = {EPOCHS}, Learning rate = {learning_rate}, Validation accuracy = {val_acc.numpy():.4f}\n")



