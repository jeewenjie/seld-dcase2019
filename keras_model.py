#
# The SELDnet architecture
#

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np

def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, weights):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    filters_list = [32,64,128]
    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=filters_list[i], kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    #spec_cnn = Permute((2, 1, 3))(spec_cnn)
    

    # More CNN
    #spec_rnn = Reshape((data_in[-2], -1))(spec_cnn)
    for j in range(5):
       for nb_kernels in [1,3,1]:
          spec_cnn = Conv2D( filters =128, kernel_size=nb_kernel, padding='same'
          )(spec_cnn) # Outputs = B, 128, 128, 4
    
    doa = spec_cnn
    sed =spec_cnn
    
    doa = Conv2D(filters=22, kernel_size = 1, padding= 'same') # B, 22, 128, 4
    sed = Conv2D(filters=11, kernel_size = 1, padding= 'same') # B, 11, 128, 4
        
    doa = Permute((3,2,1))(doa) # B, 4, 128, 22
    sed = Permute((3,2,1))(sed) # B, 4, 128, 11
    
    doa  = Conv2D(filters=1, padding='same')(doa) # B, 1, 128, 22
    sed  = Conv2D(filters=1, padding='same')(sed) # B, 1, 128, 11

    doa = Activation('linear', name='doa_out')(doa)
    sed = Activation('sigmoid', name='sed_out')(sed)
    
    doa = np.squeeze(doa,axis=1)
    sed = np.squeeze(sed,axis=1)
    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    return model
