from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Concatenate, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, Dense, AveragePooling2D
from tensorflow.keras.optimizers import SGD
import math
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model

        
class DenseNetCIFAR:
    
    def __init__(self):
        self.model = None
        
    def addTransitionLayer(self, inputs, nChannels, num_block):
        """
        Add transition layer to the model
        :param inputs: Keras Model
        :param nChannels: number of channels for the convolution
        :param num_block: number corresponding to this block
        :return: input model with transition layer added
        """
        
        x = BatchNormalization(name='transition_bn_block_' + str(num_block))(inputs)
        activ = Activation('relu', name='transition_relu_block_' + str(num_block))(x)
        x = Conv2D(nChannels, kernel_size=(1,1), padding='same', use_bias=True, kernel_regularizer=l2(0.0001), name='transition_conv_block_' + str(num_block))(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2), name='transition_pool_block_' + str(num_block))(x)
        return x
        
    def addDenseLayer(self, x, nChannels, num_layer, num_block, bottleneck=True):
        """
        Add dense layer to the model
        :param x: Keras model
        :param nChannels: number of channels for the convolution
        :param num_layer: number corresponding to this layer
        :param num_block: number corresponding to this block
        :param bottleneck: wheither to add a bottleneck layer or not (Densenet B)
        :return: input model with dense layer added
        """
        
        if bottleneck:
            x = BatchNormalization(name='bottleneck_bn_block_' + str(num_block) + '_'+ str(num_layer))(x)
            x = Activation('relu', name='bottleneck_relu_block_' + str(num_block) + '_'+ str(num_layer))(x)
            x = Conv2D(4*nChannels, kernel_size=(1,1), use_bias=True, kernel_regularizer=l2(0.0001), name='bottleneck_conv_block_' + str(num_block) + '_'+ str(num_layer))(x)
            
        x = BatchNormalization(name='dense_bn_block_' + str(num_block) + '_'+ str(num_layer))(x)
        x = Activation('relu', name='dense_relu_block_' + str(num_block) + '_'+ str(num_layer))(x)
        x = Conv2D(nChannels, kernel_size=(3,3), use_bias = True, padding='same', kernel_regularizer=l2(0.0001), name='dense_conv_block_' + str(num_block) + '_'+ str(num_layer))(x)
        return x
    
    def addDenseBlock(self, inputs, num_layers, nChannels, growthRate, num_block, bottleneck):
        """
        Add dense block to the model
        :param inputs: Keras model
        :param num_layers: number of layers for this block
        :param nChannels: number of channels for this block
        :param growthRate: growth rate at which to increase the number of channels
        :param num_block: number corresponding to this block
        :param bottleneck: wheither to add a bottleneck layer or not (Densenet B)
        :return: input model with dense block added, number of channels
        """
        
        inputs_concat = inputs
        
        for i in range(num_layers):
            inputs = self.addDenseLayer(inputs_concat, growthRate, i, num_block, bottleneck)
            inputs_concat = Concatenate()([inputs_concat, inputs])
            nChannels += growthRate
        return inputs_concat, nChannels
        
    def create_model(self, k, L, bottleneck=True):
        """
        Create model with the specified parameters
        :param k: growth rate for the network
        :param L: depth of the network
        :param bottleneck: wheither to add a bottleneck layer or not (Densenet B)
        """
        
        growthRate = k
        dropRate = 0
        nChannels = 2 * k
        reduction = 0.5
        
        N = int((L-4)/3)
        if bottleneck:
            N = int(N/2)
        
        inputs = Input(shape=(32,32,3))
        
        # Initial block (input : (224x224))
        init_conv = Conv2D(nChannels, kernel_size=(7,7), strides=(2,2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(inputs)
        
        # Block 1 (input : (56x56))
        block1, nChannels = self.addDenseBlock(inputs = init_conv, num_layers=N, nChannels=nChannels, growthRate=k, num_block=1, bottleneck=bottleneck)
        trans1 = self.addTransitionLayer(inputs=block1, nChannels=math.floor(nChannels*reduction), num_block=1)
        nChannels = math.floor(nChannels*reduction)
        
        # Block 2(input : (28x28))
        block2, nChannels = self.addDenseBlock(inputs = trans1, num_layers=N, nChannels=nChannels, growthRate=k, num_block=2, bottleneck=bottleneck)
        trans2 = self.addTransitionLayer(inputs=block2, nChannels=math.floor(nChannels*reduction), num_block=2)
        nChannels = math.floor(nChannels*reduction)
        
        # Block 3(input : (14x14))
        block3, nChannels = self.addDenseBlock(inputs = trans2, num_layers=N, nChannels=nChannels, growthRate=k, num_block=3, bottleneck=bottleneck)
        
        bn = BatchNormalization()(block3)
        activ = Activation('relu')(bn)
        pool = GlobalAveragePooling2D()(activ)
        dense = Dense(10, kernel_regularizer=l2(0.0001))(pool)
        activation=Activation('softmax')(dense)
        
        x = activation
        model = Model(inputs=inputs, outputs=x)
        sgd = SGD(lr=.1, momentum=.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        
    def step_decay(self, epoch):
        if epoch > 225:
            return 0.001
        elif epoch > 150:
            return 0.01
        else:
            return 0.1
        
    def train_model(self, train_batches, valid_batches, epochs=300, init_epoch=0):
        """
        Train model with train_batches and test it on valid_batches
        :param train_batches: generator for the training samples
        :param valid_batches: generator for the validation samples
        :param epochs: number of epochs for the training
        :param init_epoch: epoch at which to start the training (useful when resuming training)
        """
        lrate = LearningRateScheduler(self.step_decay)
        
        self.model.fit_generator(train_batches, epochs=epochs, validation_data=valid_batches, initial_epoch=init_epoch, callbacks=[lrate])
        
    def save_model(self, filename):
        """
        Save model to a file
        :param filename: name of the h5 file
        """
        self.model.save(filename)

    @staticmethod
    def load_model(filename):
        """
        Load model from a file
        :param filename: name of the h5 file
        :return: DenseNetCIFAR class with model loaded
        """
        resnet_cifar_loaded = DenseNetCIFAR()
        resnet_cifar_loaded.model = load_model(filename)

        return resnet_cifar_loaded