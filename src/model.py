# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     GlobalAveragePooling2D, Dense)
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(224, 224, 3), weight_decay=1e-4):
    """
    Deep CNN architecture for binary classification:
      - 3 blocks: [Conv → BN → ReLU] × 2 → MaxPool → Dropout
      - GlobalAveragePooling → Dense(128) → Dropout → Output
    """
    model = Sequential()

    # --- Block 1 ---
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # --- Block 2 ---
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # --- Block 3 ---
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # --- Classifier Head ---
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(128,
                    kernel_regularizer=l2(weight_decay),
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: 1 output

    return model


if __name__ == "__main__":
    m = build_model()

    import numpy as np
    random_input = np.random.random((1, 224, 224, 3))
    output = m.predict(random_input)

    print(output)
    m.summary()
