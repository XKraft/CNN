import tensorflow as tf

class MyCNN(tf.keras.Model):
    def __init__(self):
        super(MyCNN, self).__init__()
        #卷积层1
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        #BN层1
        self.b1 = tf.keras.layers.BatchNormalization()
        #激活层1
        self.a1 = tf.keras.layers.Activation('relu')
        #池化层1
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        #休眠层1
        self.d1 = tf.keras.layers.Dropout(0.1)

        #卷积层2
        self.c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same')
        #BN层2
        self.b2 = tf.keras.layers.BatchNormalization()
        #激活层2
        self.a2 = tf.keras.layers.Activation('relu')
        #池化层2
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        #休眠层2
        self.d2 = tf.keras.layers.Dropout(0.2)

        #卷积层3
        self.c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        #BN层3
        self.b3 = tf.keras.layers.BatchNormalization()
        #激活层3
        self.a3 = tf.keras.layers.Activation('relu')
        #池化层3
        self.p3 = tf.keras.layers.MaxPool2D(pool_size =(2, 2), strides=2, padding='same')
        #休眠层3
        self.d3 = tf.keras.layers.Dropout(0.2)

        #展开
        self.flatten = tf.keras.layers.Flatten()
        #FC1
        self.f1 = tf.keras.layers.Dense(4096, activation='relu')
        #休眠层4
        self.d4 = tf.keras.layers.Dropout(0.2)
        #FC2
        self.f2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d4(x)
        y = self.f2(x)
        return y