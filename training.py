import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
import json
import gc

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Paths
project_dir = os.path.abspath(os.path.dirname(__file__))
train_file = os.path.join(project_dir, "train")
labels_csv = os.path.join(project_dir, "labels.csv")
model_file = os.path.join(project_dir, "Final_dog_identification.h5")

# Parameters
im_size = 299  # Xception default
batch_size = 8
df_labels = pd.read_csv(labels_csv)

if 'breed' not in df_labels.columns or 'id' not in df_labels.columns:
    raise ValueError("The labels.csv must contain 'id' and 'breed' columns.")

df_labels["img_id"] = df_labels["id"] + ".jpg"

# Filter classes with at least 2 images
breed_counts = df_labels['breed'].value_counts()
valid_breeds = breed_counts[breed_counts >= 2].index
df_labels = df_labels[df_labels['breed'].isin(valid_breeds)]

train_df, val_df = train_test_split(df_labels, stratify=df_labels['breed'], test_size=0.2, random_state=42)

# Class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['breed']),
    y=train_df['breed']
)
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}

# Data Augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_file,
    x_col="img_id",
    y_col="breed",
    target_size=(im_size, im_size),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
)

with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_file,
    x_col="img_id",
    y_col="breed",
    target_size=(im_size, im_size),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Define model using Xception
base_model = Xception(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(df_labels['breed'].unique()), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: Freeze base
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
callbacks = [early_stopping, lr_scheduler]

# Train phase 1
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Save after Phase 1
model.save("phase_1_model.h5", include_optimizer=False)

# ✅ Clear GPU memory after Phase 1
K.clear_session()
gc.collect()

# Reload model and base (needed after clear_session)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(df_labels['breed'].unique()), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Reload weights from phase 1
model.load_weights("phase_1_model.h5")

# Phase 2: Fine-tune last few layers
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train phase 2
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Final Save
model.save(model_file, include_optimizer=False)
