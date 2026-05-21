import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =====================================
# SETTINGS
# =====================================

DATASET = r"D:\CNN_Dataset"

CSV_PATH = "D:\Milk_adulteration_project\metadata.csv"

IMG_SIZE = (224,224)

BATCH_SIZE = 16

EPOCHS = 25


# =====================================
# LOAD METADATA
# =====================================

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns)

print(df.head())


# =====================================
# STRATIFIED SPLIT
# =====================================

df["strata"] = df["class"]

train_df, val_df = train_test_split(

    df,

    test_size=0.20,

    stratify=df["strata"],

    random_state=42
)

print("\nTrain:",len(train_df))
print("Validation:",len(val_df))


# =====================================
# AUGMENTATION
# =====================================

train_datagen = ImageDataGenerator(

    preprocessing_function=
    tf.keras.applications.mobilenet_v2.preprocess_input,

    rotation_range=10,

    zoom_range=.08,

    width_shift_range=.08,

    height_shift_range=.08,

    brightness_range=[0.95,1.05],

    horizontal_flip=True
)

val_datagen = ImageDataGenerator(

    preprocessing_function=
    tf.keras.applications.mobilenet_v2.preprocess_input
)


# =====================================
# GENERATORS
# =====================================

train = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory=DATASET,

    x_col='filepath',

    y_col='class',

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True
)


val = val_datagen.flow_from_dataframe(

    dataframe=val_df,

    directory=DATASET,

    x_col='filepath',

    y_col='class',

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=False
)


# =====================================
# SAVE CLASS LABELS
# =====================================

with open(

    "class_names.json",

    "w"

) as f:

    json.dump(
        train.class_indices,
        f
    )

print("\nClass labels")
print(train.class_indices)


# =====================================
# CLASS WEIGHTS
# =====================================

labels=train.classes

weights=compute_class_weight(

    class_weight='balanced',

    classes=np.unique(labels),

    y=labels
)

class_weights=dict(
    enumerate(weights)
)

print("\nClass weights")
print(class_weights)


# =====================================
# MODEL
# =====================================

base=MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)

)

base.trainable=False


x=base.output

x=GlobalAveragePooling2D()(x)

x=BatchNormalization()(x)

x=Dropout(.35)(x)


x=Dense(

    128,

    activation='relu',

    kernel_regularizer=
    tf.keras.regularizers.l2(0.001)

)(x)


x=BatchNormalization()(x)

x=Dropout(.25)(x)


NUM_CLASSES=len(
    train.class_indices
)


output=Dense(

    NUM_CLASSES,

    activation='softmax'

)(x)


model=Model(

    base.input,

    output
)


# =====================================
# COMPILE
# =====================================

model.compile(

optimizer=tf.keras.optimizers.Adam(
learning_rate=1e-4
),

loss='categorical_crossentropy',

metrics=['accuracy']

)

model.summary()


# =====================================
# CALLBACKS
# =====================================

callbacks=[

ModelCheckpoint(

"best_model.keras",

monitor='val_accuracy',

save_best_only=True,

verbose=1

),

EarlyStopping(

monitor='val_loss',

patience=5,

restore_best_weights=True

),

ReduceLROnPlateau(

monitor='val_loss',

factor=.5,

patience=2,

verbose=1

)

]


# =====================================
# TRAIN
# =====================================

history=model.fit(

    train,

    validation_data=val,

    epochs=EPOCHS,

    callbacks=callbacks,

    class_weight=class_weights

)


# =====================================
# LOAD BEST MODEL
# =====================================

model=tf.keras.models.load_model(
    "best_model.keras"
)


model.save(
    "milk_model.keras"
)


# =====================================
# PLOTS
# =====================================

plt.figure(figsize=(12,5))


plt.subplot(1,2,1)

plt.plot(
history.history['accuracy']
)

plt.plot(
history.history['val_accuracy']
)

plt.legend(
['train','validation']
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.subplot(1,2,2)

plt.plot(
history.history['loss']
)

plt.plot(
history.history['val_loss']
)

plt.legend(
['train','validation']
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()


# =====================================
# EVALUATION
# =====================================

pred=model.predict(val)

pred=np.argmax(
pred,
axis=1
)

true=val.classes


cm=confusion_matrix(
true,
pred
)


plt.figure(figsize=(8,6))

sns.heatmap(

cm,

annot=True,

fmt='d',

xticklabels=list(
train.class_indices.keys()
),

yticklabels=list(
train.class_indices.keys()
)

)

plt.title(
"Confusion Matrix"
)

plt.xlabel(
"Predicted"
)

plt.ylabel(
"Actual"
)

plt.show()


print(

classification_report(

true,

pred,

target_names=list(
train.class_indices.keys()
)

)

)

print("\nTraining Complete")