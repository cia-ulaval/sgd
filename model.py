import tensorflow as tf

class Model:
    def __init__(self, input_dim, hidden_dims, output_dim, lr):
        self.num_samples = 6
        self.lr = lr
        self.layers = []
        self.v_layers = []
        self.weights_history = []

        prev_dim = input_dim
        for hdim in hidden_dims:
            W = tf.Variable(tf.random.normal([prev_dim, hdim], stddev=0.1))
            b = tf.Variable(tf.zeros([hdim]))
            self.layers.append((W, b))
            self.v_layers.append((tf.Variable(tf.zeros_like(W)), tf.Variable(tf.zeros_like(b))))
            self.weights_history.append([])
            prev_dim = hdim

        self.W_out = tf.Variable(tf.random.normal([prev_dim, output_dim], stddev=0.1))
        self.b_out = tf.Variable(tf.zeros([output_dim]))
        self.v_Wout = tf.Variable(tf.zeros_like(self.W_out))
        self.v_bout = tf.Variable(tf.zeros_like(self.b_out))

    def forward(self, X):
        a = X
        activations = []
        pre_activations = []

        for (W, b) in self.layers:
            z = tf.matmul(a, W) + b
            a = tf.nn.relu(z)
            pre_activations.append(z)
            activations.append(a)

        z_out = tf.matmul(a, self.W_out) + self.b_out
        y_hat = tf.nn.softmax(z_out)
        return pre_activations, activations, z_out, y_hat

    def normalSGD_train_step(self, X, y_true):
        with tf.GradientTape() as tape:
            pre_acts, acts, z_out, y_hat = self.forward(X)
            y_onehot = tf.one_hot(y_true, depth=y_hat.shape[1])
            loss = -tf.reduce_mean(tf.reduce_sum(y_onehot * tf.math.log(y_hat + 1e-9), axis=1))

        vars_all = [v for pair in self.layers for v in pair] + [self.W_out, self.b_out]
        grads = tape.gradient(loss, vars_all)

        for i, (W, b) in enumerate(self.layers):
            dW, db = grads[2 * i], grads[2 * i + 1]
            W.assign_sub(self.lr * dW)
            b.assign_sub(self.lr * db)

        self.W_out.assign_sub(self.lr * grads[-2])
        self.b_out.assign_sub(self.lr * grads[-1])

        return loss
    # Faire en sorte que l'on injecte du bruit différent pour chaque item dans la batch. Voir article benjamin
    def Laplacian_train_step_with_prop_noise(self, X, y_true):
        total_loss = 0.0
        total_grads = [[tf.zeros_like(W), tf.zeros_like(b)] for (W, b) in self.layers]
        total_grads_out = [tf.zeros_like(self.W_out), tf.zeros_like(self.b_out)]

        for i, (W, _) in enumerate(self.layers):
            self.weights_history[i].append(tf.identity(W))

        for _ in range(self.num_samples):
            with tf.GradientTape() as tape:
                a = X
                activations = []
                pre_activations = []
                noisy_layers = []
                for (W, b) in self.layers:
                    Sigma_W = tf.random.uniform(W.shape, 0.001, 0.01)
                    # reduire/augmenter variance de loi normale et tester avec un graphe des performances
                    noise_W = Sigma_W * tf.abs(W) * tf.random.normal(W.shape)
                    z = tf.matmul(a, W + noise_W) + b
                    a = tf.nn.relu(z)
                    pre_activations.append(z)
                    activations.append(a)
                    noisy_layers.append((W + noise_W, b))

                Sigma_Wout = tf.random.uniform(self.W_out.shape, 0.001, 0.01)
                noise_Wout = Sigma_Wout * tf.random.normal(self.W_out.shape)
                z_out = tf.matmul(a, self.W_out + noise_Wout) + self.b_out
                y_hat = tf.nn.softmax(z_out)

                y_onehot = tf.one_hot(y_true, depth=y_hat.shape[1])
                loss = -tf.reduce_mean(tf.reduce_sum(y_onehot * tf.math.log(y_hat + 1e-9), axis=1))

            vars_all = [v for pair in self.layers for v in pair] + [self.W_out, self.b_out]
            grads = tape.gradient(loss, vars_all)

            for i in range(len(self.layers)):
                total_grads[i][0] += grads[2 * i]
                total_grads[i][1] += grads[2 * i + 1]
            total_grads_out[0] += grads[-2]
            total_grads_out[1] += grads[-1]
            total_loss += loss

        for i, ((W, b), (vW, vb)) in enumerate(zip(self.layers, self.v_layers)):
            hist = self.weights_history[i]
            if len(hist) >= 2:
                beta = self.calculate_momentum(hist[-2], hist[-1])
            else:
                beta = 0.9

            vW.assign(beta * vW + (1 - beta) * (total_grads[i][0] / self.num_samples))
            vb.assign(beta * vb + (1 - beta) * (total_grads[i][1] / self.num_samples))
            W.assign_sub(self.lr * vW)
            b.assign_sub(self.lr * vb)

        self.v_Wout.assign(0.9 * self.v_Wout + 0.1 * (total_grads_out[0] / self.num_samples))
        self.v_bout.assign(0.9 * self.v_bout + 0.1 * (total_grads_out[1] / self.num_samples))
        self.W_out.assign_sub(self.lr * self.v_Wout)
        self.b_out.assign_sub(self.lr * self.v_bout)

        return total_loss / self.num_samples, int(self.num_samples)

    def Laplacian_train_step(self, X, y_true):
        total_loss = 0.0
        total_grads = [[tf.zeros_like(W), tf.zeros_like(b)] for (W, b) in self.layers]
        total_grads_out = [tf.zeros_like(self.W_out), tf.zeros_like(self.b_out)]

        for i, (W, _) in enumerate(self.layers):
            self.weights_history[i].append(tf.identity(W))

        for _ in range(self.num_samples):
            with tf.GradientTape() as tape:
                a = X
                activations = []
                pre_activations = []

                # ---- Add Laplacian (anisotropic) noise per layer ----
                noisy_layers = []
                for (W, b) in self.layers:
                    Sigma_W = tf.random.uniform(W.shape, 0.001, 0.01)
                    noise_W = Sigma_W * tf.random.normal(W.shape)
                    z = tf.matmul(a, W + noise_W) + b  # noisy weights
                    a = tf.nn.relu(z)
                    pre_activations.append(z)
                    activations.append(a)
                    noisy_layers.append((W + noise_W, b))

                # ---- Output layer with same type of perturbation ----
                Sigma_Wout = tf.random.uniform(self.W_out.shape, 0.001, 0.01)
                noise_Wout = Sigma_Wout * tf.random.normal(self.W_out.shape)
                z_out = tf.matmul(a, self.W_out + noise_Wout) + self.b_out
                y_hat = tf.nn.softmax(z_out)


                y_onehot = tf.one_hot(y_true, depth=y_hat.shape[1])
                loss = -tf.reduce_mean(tf.reduce_sum(y_onehot * tf.math.log(y_hat + 1e-9), axis=1))


            vars_all = [v for pair in self.layers for v in pair] + [self.W_out, self.b_out]
            grads = tape.gradient(loss, vars_all)

            for i in range(len(self.layers)):
                total_grads[i][0] += grads[2 * i]
                total_grads[i][1] += grads[2 * i + 1]
            total_grads_out[0] += grads[-2]
            total_grads_out[1] += grads[-1]
            total_loss += loss

        for i, ((W, b), (vW, vb)) in enumerate(zip(self.layers, self.v_layers)):
            hist = self.weights_history[i]
            if len(hist) >= 2:
                beta = self.calculate_momentum(hist[-2], hist[-1])
            else:
                beta = 0.9

            vW.assign(beta * vW + (1 - beta) * (total_grads[i][0] / self.num_samples))
            vb.assign(beta * vb + (1 - beta) * (total_grads[i][1] / self.num_samples))
            W.assign_sub(self.lr * vW)
            b.assign_sub(self.lr * vb)

        self.v_Wout.assign(0.9 * self.v_Wout + 0.1 * (total_grads_out[0] / self.num_samples))
        self.v_bout.assign(0.9 * self.v_bout + 0.1 * (total_grads_out[1] / self.num_samples))
        self.W_out.assign_sub(self.lr * self.v_Wout)
        self.b_out.assign_sub(self.lr * self.v_bout)

        return total_loss / self.num_samples, int(self.num_samples)

    def calculate_momentum(self, W_prev, W_curr, gamma=0.5):
        delta = tf.norm(W_curr - W_prev)
        scale = tf.norm(W_prev) + 1e-9
        moment = tf.exp(-gamma * delta / scale)
        return tf.clip_by_value(moment, 0.8, 1.0)