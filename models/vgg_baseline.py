import tensorflow as tf
from tensorflow.keras import layers, Model


class VggBaseline(Model):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1a = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bn1a = layers.BatchNormalization()
        self.conv1b = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bn1b = layers.BatchNormalization()

        self.conv2a = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.bn2a = layers.BatchNormalization()
        self.conv2b = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.bn2b = layers.BatchNormalization()

        self.conv3a = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.bn3a = layers.BatchNormalization()
        self.conv3b = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.bn3b = layers.BatchNormalization()

        self.conv4a = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.bn4a = layers.BatchNormalization()
        self.conv4b = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.bn4b = layers.BatchNormalization()

        self.pool = layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = layers.Flatten()

        self.lin1 = layers.Dense(4096, activation='relu')
        self.lin2 = layers.Dense(4096, activation='relu')
        self.lin3 = layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.bn1a(self.conv1a(x), training=training)
        x = self.bn1b(self.conv1b(x), training=training)
        x = self.pool(x)

        x = self.bn2a(self.conv2a(x), training=training)
        x = self.bn2b(self.conv2b(x), training=training)
        x = self.pool(x)

        x = self.bn3a(self.conv3a(x), training=training)
        x = self.bn3b(self.conv3b(x), training=training)
        x = self.pool(x)

        x = self.bn4a(self.conv4a(x), training=training)
        x = self.bn4b(self.conv4b(x), training=training)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x
