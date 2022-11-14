import datetime
import tensorflow as tf
import uuid


class _SoftPlus(tf.keras.layers.Layer):
    def __init__(self, beta=10.):
        tf.keras.layers.Layer.__init__(self, name='softmax')
        self._beta = beta
    def call(self, x):
        return tf.math.log(tf.ones_like(x) + tf.math.exp(x))


class _LayerSequence(tf.keras.layers.Layer):
    def __init__(self, name):
        tf.keras.layers.Layer.__init__(self, name=name)
        self._seq = []

    def call(self, features):
        out = features
        for layer in self._seq:
            out = layer(out)
        return out


class ModelUpdateCallback(tf.keras.callbacks.LambdaCallback):
    def __init__(self, model):
        tf.keras.callbacks.LambdaCallback.__init__(self, on_batch_end=self.update)
        self.model = model

    def update(self):
        """Call this method directly if you want to manually change the model's internal information. This is not
           recommended unless you know what you are doing."""
        self.model.uuid.assign(uuid.uuid4().hex)
        self.model.created_at.assign(str(datetime.datetime.utcnow()))
        self.model.global_step.assign_add(1)


class _GatedConv2D(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides, l2_reg, activation=None, depth_multiplier=3, name=''):
        _LayerSequence.__init__(self, name=name)
        if depth_multiplier is not None:
            self._seq.extend([tf.keras.layers.SeparableConv2D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                              strides=strides, depth_multiplier=depth_multiplier,
                                                              depthwise_initializer='he_uniform',
                                                              pointwise_initializer='he_uniform',
                                                              depthwise_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                              pointwise_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                              activation=activation),
                              tf.keras.layers.SeparableConv2D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                              strides=strides, depth_multiplier=depth_multiplier,
                                                              depthwise_initializer='he_uniform',
                                                              pointwise_initializer='he_uniform',
                                                              depthwise_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                              pointwise_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                              activation=tf.nn.sigmoid)])
        else:
            self._seq.extend([tf.keras.layers.Conv2D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                     strides=strides, kernel_initializer='he_uniform',
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                     activation=activation),
                              tf.keras.layers.Conv2D(filters=kernels, kernel_size=kernel_size, padding='same',
                                                     strides=strides, kernel_initializer='uniform',
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                     activation=tf.nn.sigmoid)])

    def call(self, features):
        return self._seq[0](features) * self._seq[1](features)


class _Block(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides=1, depth_multiplier=3, l2_reg=0.0, activation=None, name=''):
        _LayerSequence.__init__(self, name=name)

        # build the convolution block --
        # NOTE: Deconvolution is performed in a separate convolutional layer in order to remedy the artifacting
        #       resulting from upsampled convolutions. Performing a second convolution on its heals, ensures the
        #       artifacts are learned and undone. This is a common approach to this documented issue.
        self._seq.extend([_GatedConv2D(kernels, kernel_size, strides, l2_reg, None, depth_multiplier),
                          tf.keras.layers.BatchNormalization(),
                          activation])


class _BlockSequence(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides=1, dilates=1, depth_multiplier=3, l2_reg=0.0, activation=None, name=''):
        _LayerSequence.__init__(self, name=name)

        # build the convolution block --
        # NOTE: Deconvolution is performed in a separate convolutional layer in order to remedy the artifacting
        #       resulting from upsampled convolutions. Performing a second convolution on its heals, ensures the
        #       artifacts are learned and undone. This is a common approach to this documented issue.
        self._seq.extend([tf.keras.layers.Conv2DTranspose(filters=kernels, kernel_size=kernel_size,
                                                          padding='same', strides=dilates,
                                                          kernel_initializer='he_uniform',
                                                          bias_initializer='he_uniform',
                                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                          bias_regularizer=tf.keras.regularizers.l2(l2_reg))]
                          if dilates != 1 else
                         [_GatedConv2D(kernels, kernel_size, strides, l2_reg, None, depth_multiplier)] +
                         [tf.keras.layers.BatchNormalization(),
                          activation,
                          _GatedConv2D(kernels, 5, 1, l2_reg, None, depth_multiplier),
                          tf.keras.layers.BatchNormalization(),
                          activation,
                          _GatedConv2D(kernels, 5, 1, l2_reg, None, depth_multiplier),
                          tf.keras.layers.BatchNormalization(),
                          activation])


class _FinalBlock(_LayerSequence):

    def __init__(self, kernels, kernel_size, strides=1, depth_multiplier=3, l2_reg=0.0, name=''):
        _LayerSequence.__init__(self, name=name)

        self._seq.extend([tf.keras.layers.Conv2DTranspose(filters=kernels, kernel_size=kernel_size,
                                                          padding='same', strides=strides,
                                                          kernel_initializer='he_uniform',
                                                          bias_initializer='he_uniform',
                                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                                          activation=tf.keras.layers.LeakyReLU()),
                          _GatedConv2D(kernels, 5, 1, l2_reg, depth_multiplier=depth_multiplier,
                                       activation=tf.keras.layers.LeakyReLU())])


class _Encoder(_LayerSequence):

    NAME = 'Encoder'

    def __init__(self, kernels, kernel_size, strides=None, depth_multiplier=3, l2_reg=0.0, name=None):
        _LayerSequence.__init__(self, name=self.NAME if name is None else name)
        assert len(kernels) == len(kernel_size), 'The kernel and filter sizes must match to be valid'
        if strides is not None:
            assert len(kernels) == len(strides), 'The length of strides does not match. This is invalid'
        else:
            strides = [(1, 1) for _ in kernels]
        self._seq.extend([_GatedConv2D(k, ks, s, l2_reg, tf.keras.layers.LeakyReLU(), depth_multiplier)
                          for ii, (k, ks, s) in enumerate(zip(kernels, kernel_size, strides))])

    def call(self, features):
        self._residuals = [features]
        for layer in self._seq:
            self._residuals.append(layer(self._residuals[-1]))
        return self._residuals[::-1]


class _Decoder(_LayerSequence):

    NAME = 'Decoder'

    def __init__(self, num_channels, kernels, kernel_size, strides=None, depth_multiplier=3, l2_reg=0.0):
        _LayerSequence.__init__(self, name=self.NAME)
        assert len(kernels) == len(kernel_size), 'The kernel and filter sizes must match to be valid'
        if strides is not None:
            assert len(kernels) == len(strides), 'The length of strides does not match. This is invalid'
        else:
            strides = [(1, 1) for _ in kernels]
        self._seq.extend([_BlockSequence(k, ks, s, d, depth_multiplier, l2_reg,
                                         tf.keras.layers.LeakyReLU(), name=f'Block_{ii}')
                          for ii, (k, ks, s, d) in enumerate(zip(kernels[1:], kernel_size[:-1],
                                                                 len(strides[:-1]) * [1], strides[:-1]))])

        # add matching decode layer
        self._seq.append(_FinalBlock(kernels=kernels[-1], kernel_size=kernel_size[-1], strides=strides[-1],
                                     depth_multiplier=depth_multiplier, l2_reg=l2_reg))

        # add the pointwise squashing layer
        self.squash_layer = _GatedConv2D(num_channels, 1, 1, l2_reg, _SoftPlus(), depth_multiplier)

    # @tf.function
    def call(self, features):
        def squasher(o):
            if features[-1].shape[-1] == o.shape[-1]:
                o = tf.clip_by_value(tf.where(features[-1] == 0, o, features[-1]), 0., 1.)
            else:
                o = tf.clip_by_value(o, 0., 1.)
            return o

        out = features[0]
        for layer, residual in zip(self._seq, features[1:]):
            out = tf.concat((layer(out), residual), axis=-1)

        return squasher(self.squash_layer(out))


class UNet(tf.keras.Model, tf.Module):
    NAME = 'UNET_GENERATOR'

    def __init__(self, options):
        tf.keras.Model.__init__(self, name=self.NAME)

        self.optimizer = tf.optimizers.RMSprop(options.base_lr, momentum=options.momentum)
        self._seq = []
        self.global_step = tf.Variable(name='global_step', shape=[], dtype=tf.int64, trainable=False, initial_value=0)
        self.uuid = tf.Variable(name='uuid', shape=[], dtype=tf.string, trainable=False,
                                initial_value=uuid.uuid4().hex)
        self.created_at = tf.Variable(name='created_at', shape=[], dtype=tf.string, trainable=False,
                                      initial_value=str(datetime.datetime.utcnow()))

        self.num_channels = tf.constant(name='num_channels', shape=[], dtype=tf.uint32, value=len(options.out_channels))
        self._encoders = [_Encoder(kernels=options.kernels,
                                   kernel_size=options.kernel_size,
                                   strides=options.strides,
                                   depth_multiplier=options.depth_multiplier,
                                   l2_reg=options.l2_reg) for _ in options.in_channels]
        self._decoder = _Decoder(num_channels=len(options.out_channels),
                                 kernels=options.kernels[::-1],
                                 kernel_size=options.kernel_size[::-1],
                                 strides=options.strides[::-1],
                                 depth_multiplier=options.depth_multiplier,
                                 l2_reg=options.l2_reg)
        self._seq.extend(self._encoders + [self._decoder])
        self._set_inputs(tf.keras.Input(shape=(None, None, len(options.in_channels)),
                                        batch_size=options.batch_size,
                                        dtype=tf.float32))
        self.build(input_shape=(None, None, None, len(options.in_channels)))
        self.summary()

    def call(self, features):
        # subsequent channels are added to one another before proceeding --
        # this diverges from some papers where some channels are added and some are concatenated
        encoder_out = [encoder(tf.stack([tf.unstack(features, axis=-1)[chan]], axis=-1))
                       for chan, encoder in zip(range(features.shape[-1]), self._encoders.layers)]
        out = encoder_out[0]
        for x2_out in encoder_out[1:]:
            out = [tf.concat((x1, x2), axis=-1) for x1, x2 in zip(out, x2_out)]
        return self._decoder(out)
