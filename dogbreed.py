import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Dynamic path configuration
project_dir = os.path.abspath(os.path.dirname(__file__))
train_file = os.path.join(project_dir, "train")
test_file = os.path.join(project_dir, "test")
labels_csv = os.path.join(project_dir, "labels.csv")
model_file = os.path.join(project_dir, "Final_dog_identification.h5")

# Parameters
im_size = 224
batch_size = 32
df_labels = pd.read_csv(labels_csv)

if 'breed' not in df_labels.columns or 'id' not in df_labels.columns:
    raise ValueError("The labels.csv must contain 'id' and 'breed' columns.")

df_labels["img_id"] = df_labels["id"] + ".jpg"

# Filter out classes with fewer than 2 images (required for stratified split)
breed_counts = df_labels['breed'].value_counts()
valid_breeds = breed_counts[breed_counts >= 2].index
df_labels = df_labels[df_labels['breed'].isin(valid_breeds)]

# Now perform stratified split
train_df, val_df = train_test_split(df_labels, stratify=df_labels['breed'], test_size=0.2, random_state=42)


# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['breed']),
    y=train_df['breed']
)
class_weights_dict = dict(enumerate(class_weights))

# Data augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
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

# Model definition
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(df_labels['breed'].unique()), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Save model
model.save(model_file)
