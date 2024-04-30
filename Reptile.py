import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Load data
data_path = 'E:\\pythonProject\\paper\\model_data.xlsx'
# data_path = 'E:\\pythonProject\\paper\\scenario_data.xlsx'
data = pd.read_excel(data_path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

# Parameter grid
neurons = [20,25,30]  # Increase number of neurons
learning_rates = [0.0001]
epochs = [2000,2500,3000,4000]  # Increase number of training rounds
batch_sizes = [1,3,5,7,10] # Reduce batch size to increase update frequency and model stability

# Grid search
best_accuracy = 0
best_parameters = {}

for neuron in neurons:
    for lr in learning_rates:
        for epoch in epochs:
            for batch_size in batch_sizes:
                print(f"Testing with {neuron} neurons, learning rate {lr}, {epoch} epochs, batch size {batch_size}")

                model = Sequential([
                    Dense(neuron, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
                    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
                ])

                optimizer = Adam(learning_rate=lr)
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                for _ in range(10):  # Repeated training for stability
                    for e in range(epoch):
                        weights_before = model.get_weights()
                        indices = np.random.randint(low=0, high=len(X_train), size=batch_size)
                        X_batch = tf.convert_to_tensor(X_train[indices], dtype=tf.float32)
                        y_batch = y_train[indices]

                        with tf.GradientTape() as tape:
                            preds = model(X_batch, training=True)
                            y_batch = tf.expand_dims(y_batch, axis=-1)
                            loss = tf.keras.losses.binary_crossentropy(y_batch, preds)
                        gradients = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                        weights_after = model.get_weights()
                        updated_weights = [0.5 * np.array(w) + 0.5 * np.array(w_before) for w, w_before in
                                           zip(weights_after, weights_before)]
                        model.set_weights(updated_weights)

                final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
                if final_accuracy > best_accuracy:
                    best_accuracy = final_accuracy
                    best_parameters = {'neurons': neuron, 'learning_rate': lr, 'epochs': epoch,
                                       'batch_size': batch_size}
                    print(f"New best accuracy: {best_accuracy} with parameters {best_parameters}")

print(f"Best accuracy achieved: {best_accuracy} with parameters {best_parameters}")
