import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
img_size = 224  # Adjust based on VGG16 input size
batch_size = 32
num_classes = 9

data_dir = 'Dataset'

# Data preprocessing and augmentation (reduced due to VGG16 pre-processing)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

# Load and augment training data (use flow_from_directory with target_size matching VGG16)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data (use flow_from_directory with target_size matching VGG16)
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load VGG16 model with pre-trained weights and include top (modified for 9 classes)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# Freeze convolutional layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=25,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
print(f'Validation accuracy: {accuracy}')

# Save the model
model_path = 'Trained_Model/CNN_Model_VGG16_v2.keras'
model.save(model_path)

print(model.summary())
