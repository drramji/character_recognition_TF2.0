import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

## if tensorflow_datasets dataset is not insalled, first install it as follow
# pip install tensorflow-datasets
# or
# conda install tensorflow-datasets

## 1. Prepare Data
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

print("\n...Data Preparation Completed...")

input_size = 784
output_size = 10
hidden_layer_size = 40

## Create a Model
print("\n...Model Creation started...")
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

print('input_size: ', input_size)

print("\n...Model Creation Completed...")

## Choose the optimizer and the loss function
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\n...optimizer and loss fn that is optimization function are set...")

# print(model)

## Training the Model
NUM_EPOCHS = 2 #5
print("\n...Training started...")
# model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2, validation_steps=10)
model.fit(train_data, epochs = NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2, steps_per_epoch = 500, validation_steps=10)
print("\n...Training Ended...")

## Test the model
print("\n...Now test the model ...")
test_loss, test_accuracy = model.evaluate(test_data)

print("\n\n..............Final Test Results are :................\n")
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy * 100))