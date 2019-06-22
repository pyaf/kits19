from functools import partial
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D, Conv3DTranspose,AveragePooling3D
from keras.engine import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization 
from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss,dice_coefficient_loss
from keras import backend as K
from keras.layers import Activation, add, multiply, Lambda

kinit = 'glorot_normal'
K.set_image_data_format('channels_first') # TF dimension ordering in this code
def UnetConv3D(input, outdim, is_batchnorm, name):
    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name+'_1', data_format = 'channels_first')(input)
    if is_batchnorm:
        x =BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu',name=name + '_1_act')(x)

    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name+'_2', data_format = 'channels_first')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x
    
def unet(input_shape=(4, 128, 128, 128), n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss):   
  
    inputs = Input(shape=input_shape)
    conv1 = UnetConv3D(inputs, 32, is_batchnorm=False, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2,2 ))(conv1)
    
    conv2 = UnetConv3D(pool1, 64, is_batchnorm=False, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2,2 ))(conv2)

    conv3 = UnetConv3D(pool2, 128, is_batchnorm=False, name='conv3')
    pool3 = MaxPooling3D(pool_size=(2, 2,2 ))(conv3)

    conv4 = UnetConv3D(pool3, 256, is_batchnorm=False, name='conv4')
    pool4 = MaxPooling3D(pool_size=(2, 2,2 ))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same', data_format = 'channels_first')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same', data_format = 'channels_first')(conv5)

    up6 = concatenate([Conv3DTranspose(256, (2, 2,2 ), strides=(2, 2,2 ), kernel_initializer=kinit, padding='same', data_format = 'channels_first')(conv5), conv4], axis=1)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', data_format = 'channels_first')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', data_format = 'channels_first')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2,2 ), strides=(2, 2,2 ), padding='same', data_format = 'channels_first')(conv6), conv3], axis=1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same', data_format = 'channels_first')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same', data_format = 'channels_first')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2,2 ), strides=(2,2,2 ), kernel_initializer=kinit, padding='same', data_format = 'channels_first')(conv7), conv2], axis=1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same', data_format = 'channels_first')(up8)

    up9    = concatenate([Conv3DTranspose(32, (2, 2,2 ), strides=(2, 2,2 ), kernel_initializer=kinit, padding='same', data_format = 'channels_first')(conv8), conv1], axis=1)
    conv9  = Conv3D(32, (3, 3, 3), activation='relu',  kernel_initializer=kinit, padding='same', data_format = 'channels_first')(up9)
    conv9  = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same', data_format = 'channels_first')(conv9)
    conv10 = Conv3D(3, (1, 1, 1), activation='relu', kernel_initializer=kinit,padding = 'same', name='final', data_format = 'channels_first')(conv9)

    activation_name = 'sigmoid'
    activation_block = Activation(activation_name)(conv10)
    model = Model(inputs=[inputs], outputs=[activation_block])
    model.compile(optimizer=optimizer(), loss=loss_function)
    return model

def expend_as(tensor, rep,name):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same', name='xl'+name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    phi_g = Conv3D(inter_shape, (1, 1,1), padding='same')(g)
    upsample_g = Conv3DTranspose(inter_shape, (3, 3,3),strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3],shape_theta_x[4] // shape_g[4]),padding='same', name='g_up'+name)(phi_g)  # 16
    # upsample_g = phi_g
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv3D(1, (1, 1,1), padding='same', name='psi'+name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling3D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3],shape_x[4] // shape_sigmoid[4]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[1],  name)
    y = multiply([upsample_psi, x], name='q_attn'+name)

    result = Conv3D(32, (1, 1,1), padding='same',name='q_attn_conv'+name)(y)
    # result = Conv3D(shape_x[1], (1, 1,1), padding='same',name='q_attn_conv'+name)(y)
    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
    return result_bn

def UnetConv3D(input, outdim, is_batchnorm, name):
    x = Conv3D(outdim, (3, 3,3), strides=(1, 1,1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
    if is_batchnorm:
        x =BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu',name=name + '_1_act')(x)

    x = Conv3D(outdim, (3, 3,3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x
    

def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = Conv3D(shape[1] * 1, (1, 1,1), strides=(1, 1,1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name = name + '_act')(x)
    return x

def attn_unet(input_shape=(4, 128, 128, 128), n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss):   
    inputs = Input(shape=input_shape)
    conv1 = UnetConv3D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = UnetConv3D(pool1, 32, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = UnetConv3D(pool2, 64, is_batchnorm=True, name='conv3')
    #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = UnetConv3D(pool3, 64, is_batchnorm=True, name='conv4')
    #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    center = UnetConv3D(pool4, 128, is_batchnorm=True, name='center')
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv3DTranspose(32, (3,3, 3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1', axis=1)
    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv3DTranspose(32, (3,3, 3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2', axis= 1)

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv3DTranspose(32, (3,3, 3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3', axis =1)

    up4 = concatenate([Conv3DTranspose(32, (3,3, 3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4', axis =1)
    out = Conv3D(1, (1, 1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up4)
    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=optimizer(lr = initial_learning_rate), loss=loss_function)
    return model


def attn_reg_ds(input_shape=(4, 128, 128, 128), n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss):
    img_input = Input(shape=input_shape, name='input_scale1')

    conv1 = UnetConv3D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2,2))(conv1)
    
    conv2 = UnetConv3D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2,2))(conv2)

    conv3 = UnetConv3D(pool2, 128, is_batchnorm=True, name='conv3')
    #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2))(conv3)
    
    conv4 = UnetConv3D(pool3, 64, is_batchnorm=True, name='conv4')
    #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2,2))(conv4)
        
    center = UnetConv3D(pool4, 512, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1', axis = 1)

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2', axis =1)

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3', axis= 1)

    up4 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4', axis =1)
    
    conv6 = UnetConv3D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv3D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv3D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv3D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv3D(1, (1, 1,1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv3D(1, (1, 1,1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv3D(1, (1, 1,1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv3D(1, (1, 1,1), activation='sigmoid', name='final')(conv9)
    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
    loss = {'pred1':loss_function,
            'pred2':loss_function,
            'pred3':loss_function,
            'final': loss_function}
    
    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    model.compile(optimizer=optimizer(lr = initial_learning_rate), loss=loss)   
    return model

def attn_reg(input_shape=(4, 128, 128, 128), n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss):
    img_input = Input(shape=input_shape, name='input_scale1')
    scale_img_2 = AveragePooling3D(pool_size=(2, 2,2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling3D(pool_size=(2, 2,2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling3D(pool_size=(2, 2,2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv3D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2,2))(conv1)
    
    input2 = Conv3D(64, (3, 3,3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=1)
    conv2 = UnetConv3D(input2, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2,2))(conv2)
    
    input3 = Conv3D(128, (3, 3,3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=1)
    conv3 = UnetConv3D(input3, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling3D(pool_size=(2, 2,2))(conv3)
    
    input4 = Conv3D(256, (3, 3,3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=1)
    conv4 = UnetConv3D(input4, 64, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling3D(pool_size=(2, 2,2))(conv4)
    center = UnetConv3D(pool4, 512, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1', axis =1)

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2', axis = 1)

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3', axis =1)

    up4 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4', axis =1)
    
    conv6 = UnetConv3D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv3D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv3D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv3D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9]) 
    loss = {'pred1':loss_function,
            'pred2':loss_function,
            'pred3':loss_function,
            'final': loss_function}
    
    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    # model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)
    model.compile(optimizer=optimizer(lr = initial_learning_rate), loss=loss)
    return model