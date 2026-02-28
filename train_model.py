import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def f05_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    beta_sq = 0.5**2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + tf.keras.backend.epsilon())

def build_v15_architecture():
    base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Lambda(preprocess_input)(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Advanced Forensic Head
    res = layers.Dense(256, activation='relu')(x)
    res = layers.BatchNormalization()(res)
    res = layers.Dropout(0.5)(res)
    
    outputs = layers.Dense(1, activation='sigmoid')(res)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy', f05_score])
    return model, base_model

if __name__ == "__main__":
    model, base_model = build_v15_architecture()
    
    # Phase 2: Fine-Tuning v15 Strategy
    base_model.trainable = True
    for layer in base_model.layers[:-50]: # Unfreeze deeper for v15
        layer.trainable = False

    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    
    model.compile(optimizer=optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=[f05_score])
    model.save("app/seed_model_v15.h5")
    print("V15 Engine Initialized and Exported.")
