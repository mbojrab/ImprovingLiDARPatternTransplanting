import multiprocessing as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.ops import array_ops

import command_line

class Pipeline:
    
    def _add_augmentation(self, example, is_training):
        """Parsing and augmenation function to run each tf.Example through to obtain the input

        Parameters
        ----------
        example : tf.data.Example
            Unserialized example from the tfrecord
        is_training : bool
            True if this is a training run, false if not

        Returns
        -------
        x, y : tuple
            x : array
                The input to the training session
            y : array
                The target for the training session
        """
        raise NotImplemented()

    def create(self, data_dir, mode, batch_size=128, num_epochs=1):
        """Create the data pipeline with the provided augmentation pipeline.

        Parameters
        ----------
        data_dir : str
            Parent directory where the .tfrecords are sitting.
        mode : str
            This is which of the files to select.
        batch_size : int
            Size of the batch to prepare.
        num_epochs : int
            Number of epochs this session will provide. The generator will be setup to repeat data after each epoch is
            achieved and throw an exception once it has provided the requested number of examples. By default only a
            single epoch is configured.

        Returns
        -------
        dataset : tf.data.Dataset
            The dataset representing the fully configured data pipeline.
        """
        # ingest the data in a threaded pipeline
        is_training = mode.lower() == command_line.TRAIN
        tf_file = os.path.join(data_dir, f'{mode}.tfrecords')

        # verify the file was previously generated
        if not os.path.exists(tf_file):
            raise Exception(f'Data Pipeline not found for [{tf_file}].')

        dataset = tf.data.TFRecordDataset(tf_file)
        dataset = dataset.map(lambda x: self._add_augmentation(x, is_training),
                              num_parallel_calls=mp.cpu_count() - 2)

        # more stochasticity and all ;)
        if is_training:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)

        # prepare one batch -- for a given number of epochs
        dataset = dataset.repeat(1 if num_epochs < 1 else int(np.ceil(num_epochs)))
        dataset = dataset.batch(batch_size)

        # prepare several batches ahead of schedule -- this keeps the beast well-fed
        dataset = dataset.prefetch(np.maximum(1, mp.cpu_count() - 2))

        return dataset


class GeneratorPipeline(Pipeline):
    """Class defining the ingest and augmentation general generative models

    Parameters
    ----------
    target_shape : np.array, shape=(3,)
        The shape to crop from within each example. This should include the fully channel width. The swizzled channels
        happens later.
    channels : np.array, shape=(n,)
        The order of the channel stack. This is equivalent to x[:, :, channels]
    """

    def __init__(self, target_shape=[300, 300, 2], channels=[0, 1], random_crop=True):
        Pipeline.__init__(self)
        self._random_crop = random_crop
        self._target_shape = target_shape
        self._channels = channels

    def _parse_example(self, example, is_training):

        # TODO: Add the ability to optionally load a target image from the example if it exists
        # prepare the items to extract from the examples
        features = {'image': tf.io.FixedLenFeature([], tf.string)}

        # extract the data from the protobuf
        doc = tf.io.parse_single_example(example, features)

        # prepare the image and record the shape of the image --
        # NOTE: it is assumed the input imagery is larger than the target shape however it is not verified
        doc['image'] = tf.io.parse_tensor(doc['image'], out_type=np.uint8)
        if is_training or self._random_crop:
            doc['image'] = tf.image.random_crop(doc['image'], self._target_shape)
        else:
            shape = tf.cast(array_ops.shape(doc['image'])[:2], tf.float32)
            offset = tf.cast((shape - self._target_shape[:2]) / 2., tf.int32)
            doc['image'] = tf.image.crop_to_bounding_box(doc['image'],
                                                         offset_height=offset[0],
                                                         offset_width=offset[1],
                                                         target_height=self._target_shape[0],
                                                         target_width=self._target_shape[1])
            doc['image'].set_shape(self._target_shape)

        # select the channels requested and in the order requested
        channels = tf.unstack(doc['image'], axis=-1)
        doc['image'] = tf.stack([channels[ii] for ii in self._channels], axis=-1)

        return doc

    def _add_augmentation(self, example, is_training):
        # parse the data
        doc = self._parse_example(example=example, is_training=is_training)

        # the original image
        x = doc['image']

        # perform data augmentation in training stages
        if is_training:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.random_brightness(x, 0.3)
            x = tf.image.random_contrast(x, 0.6, 1.4)

        x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.clip_by_value(x, 0, 1)
        y = x

        return x, y


class MaskPipeline(Pipeline):
    """Class defining the ingest and augmentation of masks
    """
    def __init__(self, target_shape=[300, 300, 1]):
        Pipeline.__init__(self)
        self._target_shape = target_shape

    def _parse_example(self, example):
        # prepare the items to extract from the examples
        features = {'image': tf.io.FixedLenFeature([], tf.string)}

        # extract the data from the protobuf
        doc = tf.io.parse_single_example(example, features)
        doc['image'] = tf.image.random_crop(tf.io.parse_tensor(doc['image'], out_type=np.uint8), self._target_shape)
        return doc

    def _add_augmentation(self, example, is_training):
        # parse the data
        doc = self._parse_example(example=example)

        # the original image
        x = doc['image']

        # perform data augmentation in training stages
        if is_training:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)

        # NOTE: the values are stores as boolean, so we directly cast back to float32
        x = tf.cast(x, tf.float32)
        y = x

        return x, y


class RandomMaskPipeline(MaskPipeline):
    """Class defining the ingest and augmentation of masks
    """

    def __init__(self, target_shape=[300, 300, 1]):
        MaskPipeline.__init__(self, target_shape)
        self._num_pixels = np.prod(target_shape)
        self._rng = tf.random.Generator.from_non_deterministic_state()

    def _add_augmentation(self, example, is_training):
        # parse the data
        doc = self._parse_example(example=example)

        # during training randomly generate a mask based on the number of pixels dropped in the real mask
        if is_training:

            # determine the percentage of active pixels in the mask --
            # this becomes the probability of an active pixel
            prob = tf.reduce_sum(doc['image']) / self._num_pixels

            num_active = tf.cast(tf.reduce_sum(tf.cast(doc['image'], dtype=tf.float32)), dtype=tf.int32)

            # generate random mask based on number of active pixels in the real mask
            rand_ii, _ = tf.unique(tf.random.uniform(shape=[num_active], minval=0, maxval=self._num_pixels, dtype=tf.int64))
            mask = tf.sparse.to_dense(tf.sparse.SparseTensor(tf.reshape(tf.sort(rand_ii), (-1,1)),
                                                             tf.ones_like(rand_ii), [self._num_pixels]), 0)
        # in validation use the real mask
        else:
            mask = tf.cast(doc['image'], tf.float32)

        x = tf.cast(tf.reshape(mask, shape=self._target_shape), dtype=tf.float32)
        y = x

        return x, y


class UnifiedDataset:
    """Class defining the automated masking of the input based on a second data pipeline.

    Here create() returns a generator object instead of a tf.data.Dataset to allow us to inject a masking into the
    pipeline that was generated from a second pipeline.

    Parameters
    ----------
    gen_dataset : tf.data.Dataset
        The dataset pipeline for the input data.
    mask_dataset : tf.data.Dataset
        The dataset pipeline for the mask data.
    """
    def __init__(self, gen_dataset, mask_dataset):
        self.gen_iter = iter(gen_dataset)
        self.mask_iter = iter(mask_dataset)

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y = next(self.gen_iter)
        mask_x, _ = next(self.mask_iter)

        # apply the mask to degrade the input image
        batch_x *= mask_x[:batch_x.shape[0]]

        return batch_x, batch_y


class UnifiedGeneratorPipeline(Pipeline):
    """Class defining the automated masking of the input based on a second data pipeline.

    Here create() returns a generator object instead of a tf.data.Dataset to allow us to inject a masking into the
    pipeline that was generated from a second pipeline.

    Parameters
    ----------
    target_shape : np.array, shape=(3,)
        The shape to crop from within each example. This should include the fully channel width. The swizzled channels
        happens later.
    channels : np.array, shape=(n,)
        The order of the channel stack. This is equivalent to x[:, :, channels]
    random_mask : bool
        Generate a random mask with the same dropout as the supplied mask
    """
    def __init__(self, target_shape=[300, 300, 2], channels=[0, 1], random_mask=False):
        self.gen_pipeline = GeneratorPipeline(target_shape=target_shape, channels=channels, random_crop=False)
        self.mask_pipeline = RandomMaskPipeline(target_shape=[target_shape[0], target_shape[1], 1]) if random_mask else\
                             MaskPipeline(target_shape=[target_shape[0], target_shape[1], 1])

    def create(self, data_dir, mode, batch_size=128, num_epochs=1):
        """Create the data pipeline with the provided augmentation pipeline.

        Parameters
        ----------
        data_dir : str
            Parent directory where the .tfrecords are sitting.
        mode : str
            This is which of the files to select.
        batch_size : int
            Size of the batch to prepare.
        num_epochs : int
            Number of epochs this session will provide. The generator will be setup to repeat data after each epoch is
            achieved and throw an exception once it has provided the requested number of examples. By default only a
            single epoch is configured.

        Returns
        -------
        dataset : tf.data.Dataset
            The dataset representing the fully configured data pipeline.
        """
        return UnifiedDataset(self.gen_pipeline.create(data_dir, mode, batch_size, num_epochs),
                              self.mask_pipeline.create(data_dir, command_line.MASKS, batch_size,
                                                        num_epochs * 10))


class GeneratorTestPipeline(Pipeline):
    """Class defining the ingest and augmentation general generative models for its test set"""

    def __init__(self):
        Pipeline.__init__(self)

    def _parse_example(self, example):

        # TODO: Add the ability to optionally load a target image from the example if it exists
        # prepare the items to extract from the examples
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'original': tf.io.FixedLenFeature([], tf.string)}

        # extract the data from the protobuf
        doc = tf.io.parse_single_example(example, features)

        # prepare the image and record the shape of the image --
        # NOTE: it is assumed the input imagery is larger than the target shape however it is not verified
        doc['image'] = tf.io.parse_tensor(doc['image'], out_type=np.uint8)
        doc['original'] = tf.io.parse_tensor(doc['original'], out_type=np.uint8)

        return doc

    def _add_augmentation(self, example, is_training):
        # parse the data
        doc = self._parse_example(example=example)

        # the original image
        def convert(x):
            x = tf.image.convert_image_dtype(x, tf.float32)
            x = tf.clip_by_value(x, 0, 1)
            return x

        return convert(doc['image']), convert(doc['original'])
