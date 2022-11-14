import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

import command_line
import pipeline
import unet
import train


def build_discriminator():
    tkl = tf.keras.layers
    l2_reg = 5e-5
    args = {'padding': 'same',
            'kernel_initializer': 'he_uniform',
            'kernel_regularizer': tf.keras.regularizers.l2(l2_reg)}
    model = tf.keras.models.Sequential([
        tkl.Conv2D(64, 3, 2, **args),
        tkl.BatchNormalization(),
        tkl.LeakyReLU(),
        tkl.Conv2D(128, 3, 2, **args),
        tkl.BatchNormalization(),
        tkl.LeakyReLU(),
        tkl.Conv2D(128, 3, 2, **args),
        tkl.GlobalAveragePooling2D(),
        tkl.Dense(1, 'sigmoid', kernel_initializer='uniform', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM))
    return model


if __name__ == '__main__':
    options = command_line.generator_options()
    options.kernel_size = [(k, 3) for k in options.kernel_size]

    # create the model
    tf.summary.trace_on()
    if os.path.exists(os.path.join(options.model_dir, 'saved_model.pb')):
        print('Loading previously saved model.')
        generator = tf.keras.models.load_model(options.model_dir)
    else:
        print('Creating a new model.')
        generator = unet.UNet(options=options)
        generator.compile(optimizer=generator.optimizer, loss='mean_absolute_error')

    discriminator = None
    if options.is_gan:
        model_dir = os.path.join(options.model_dir, 'discriminator')
        discriminator = tf.keras.models.load_model(model_dir) if os.path.exists(model_dir) else build_discriminator()

    # train the model to completion
    pipeline = pipeline.UnifiedGeneratorPipeline(target_shape=[192, 192, 2],
                                                 channels=options.in_channels,
                                                 random_mask=options.random_mask)

    # choose the appropriate training routine
    train.train_model(generator, discriminator, pipeline, options, tf.losses.Huber(delta=10))
