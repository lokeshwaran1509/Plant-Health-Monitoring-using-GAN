import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow.lite as tflite
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
Z_DIM = 100  # Latent space dimension for GAN

# Dataset path
dataset_path = r"C:\Users\Lokeshwaran\Documents\plant_disease_detection\plant_disease\Weed_dataset"  # Folder containing 'Weed' and 'No_Weed' subfolders

# Load dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training')

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation')

# Define Generator Model
def build_generator(z_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=z_dim),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(np.prod(IMG_SIZE + (3,)), activation='tanh'),
        tf.keras.layers.Reshape(IMG_SIZE + (3,))
    ])
    return model

# Define Discriminator Model
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

discriminator = build_discriminator(IMG_SIZE + (3,))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(Z_DIM)
discriminator.trainable = False

gan_input = tf.keras.layers.Input(shape=(Z_DIM,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train GAN
def train_gan(epochs, batch_size, z_dim, train_generator, discriminator, gan, generator):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        real_images, _ = next(train_generator)
        real_images = real_images[:half_batch]
        noise = np.random.normal(0, 1, (half_batch, z_dim))
        fake_images = generator.predict(noise)
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)
        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

train_gan(epochs=50, batch_size=32, z_dim=Z_DIM, train_generator=train_generator, 
          discriminator=discriminator, gan=gan, generator=generator)

# Build CNN Classifier
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

classifier = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save model
classifier.save("weed_detection_model.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
tflite_model = converter.convert()

# Save TFLite model
with open("weed_detection_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model conversion complete! TFLite model saved as weed_detection_model.tflite")

# Save the GAN model
generator.save('generator.keras')
discriminator.save('discriminator.keras')
