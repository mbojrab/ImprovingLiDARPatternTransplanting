import os
import numpy as np
import tensorflow as tf
import tqdm

import command_line
import unet


class ValidationBoardCallback:
    def __init__(self, options, randomize_image_reporting=False, max_outputs=5,
                 show_target=False, show_prediction=False):
        self.max_outputs = max_outputs
        self.show_target = show_target
        self.show_prediction = show_prediction
        num_val_batches = int(options.num_val_examples // options.batch_size)
        self.random_steps = np.random.randint(0, num_val_batches, options.max_epochs) \
                            if randomize_image_reporting else np.zeros(options.max_epochs)

    def __call__(self, x, y, pred, step, epoch, name=''):
        # add examples images to the tensorboard
        if step == self.random_steps[epoch]:
            tf.summary.image(f'{name}input', x, max_outputs=self.max_outputs, step=epoch)
            if self.show_target:
                tf.summary.image(f'{name}target', y, max_outputs=self.max_outputs, step=epoch)
            if self.show_prediction:
                tf.summary.image(f'{name}prediction', pred, max_outputs=self.max_outputs, step=epoch)


def train_model(generator, discriminator, pipeline, options, loss_func):

    board_callback = ValidationBoardCallback(options, randomize_image_reporting=True, show_target=True,
                                             show_prediction=True)
    generator_updater = unet.ModelUpdateCallback(generator)
    start_epoch = int((tf.keras.backend.get_value(generator.global_step) * options.batch_size) // options.num_train_examples)
    writer = tf.summary.create_file_writer(options.model_dir)
    global_loss = np.inf
    num_worse = 0

    plateau_cnt = 0
    reduce_on_plateau = 5

    with writer.as_default():
        with tf.summary.record_if(True):

            # output the model graph to the tensorboard
            tf.summary.trace_export(name="graph", step=0)

            # loop through the epochs
            for epoch in range(start_epoch, options.max_epochs):

                print(f'Starting Epoch [{epoch}]')

                # load the training and validation datasets
                train_data = pipeline.create(options.directory, command_line.TRAIN, options.batch_size, options.epochs)
                val_data = pipeline.create(options.directory, command_line.VALIDATION, options.batch_size, 1)

                # train the batch
                if discriminator is not None:
                    train_sum_acc = 0.
                    train_tot_acc = 0
                num_train_batches = int(options.num_train_examples // options.batch_size)
                for train_x, train_y in tqdm.tqdm(train_data, total=num_train_batches):
                    if len(options.out_channels) != train_y.shape[-1]:
                        channels = tf.unstack(train_y, axis=-1)
                        train_y = tf.stack([channels[c] for c in options.out_channels], axis=-1)

                    gst = tf.keras.backend.get_value(generator.global_step)

                    # fit the model for this batch
                    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

                        # track the gradients through the forward-pass
                        generated_y = generator(train_x, training=True)
                        if discriminator is not None:
                            real_y = discriminator(train_y, training=True)
                            pred_y = discriminator(generated_y, training=True)
                        if gst == 1:
                            generator.summary()

                        # calculate the generator loss
                        reconstruction_loss = tf.stack([ww * loss_func(train_y[:,:,:,ii:ii+1], generated_y[:,:,:,ii:ii+1])
                                                         for ww, ii in zip([1., 1.], range(train_y.shape[-1]))], axis=-1)
                        regularization_loss = tf.reduce_mean(generator.losses)
                        generator_losses = reconstruction_loss + regularization_loss
                        if discriminator is not None:
                            entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                            confuse_loss = entropy(tf.ones_like(pred_y), pred_y)
                            tf.summary.scalar('generator loss: confuse', confuse_loss, step=gst)
                            generator_losses += confuse_loss * 2e-5

                        # add the losses to the tensorboard
                        tf.summary.scalar('generator loss: reconstruction', tf.reduce_mean(reconstruction_loss), step=gst)
                        tf.summary.scalar('generator loss: regularization', regularization_loss, step=gst)
                        tf.summary.scalar('generator loss: train', tf.reduce_mean(generator_losses), step=gst)

                        # update generator
                        generator_gradients = generator_tape.gradient(generator_losses, generator.trainable_variables)
                        generator.optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

                        # update the discriminator
                        if discriminator is not None:
                            entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                            confuse_loss = entropy(tf.ones_like(real_y), real_y) + \
                                           entropy(tf.zeros_like(pred_y), pred_y)
                            regularization_loss = tf.reduce_mean(discriminator.losses)
                            discriminator_losses = confuse_loss + regularization_loss

                            # update discriminator
                            discriminator_gradients = discriminator_tape.gradient(discriminator_losses,
                                                                                  discriminator.trainable_variables)
                            discriminator.optimizer.apply_gradients(zip(discriminator_gradients,
                                                                        discriminator.trainable_variables))

                            # add the losses to the tensorboard
                            tf.summary.scalar('discriminator loss: regularization', regularization_loss, step=gst)
                            tf.summary.scalar('discriminator loss: train', discriminator_losses, step=gst)

                            train_sum_acc += tf.reduce_sum(tf.round(pred_y)).numpy()
                            train_tot_acc += pred_y.shape[0]

                    # update the model identification
                    generator_updater.update()

                if discriminator is not None:
                    tf.summary.scalar('accuracy: train', train_sum_acc / train_tot_acc, step=epoch)

                # test the model updates after the epoch
                if discriminator is not None:
                    val_sum_acc = 0.
                    val_tot_acc = 0
                val_accum_loss = 0.
                num_val_batches = int(options.num_val_examples // options.batch_size)
                mae_i = tf.keras.metrics.MeanAbsoluteError()
                mae_d = tf.keras.metrics.MeanAbsoluteError()
                rmse_i = tf.keras.metrics.RootMeanSquaredError()
                rmse_d = tf.keras.metrics.RootMeanSquaredError()
                for step, (val_x, val_y) in tqdm.tqdm(enumerate(val_data), total=num_val_batches):
                    if len(options.out_channels) != val_y.shape[-1]:
                        channels = tf.unstack(val_y, axis=-1)
                        val_y = tf.stack([channels[c] for c in options.out_channels], axis=-1)

                    # mask the generated values according to the target
                    generated_y = generator(val_x, training=False) * tf.where(val_y == 0., 0., 1.)
                    if discriminator is not None:
                        pred_y = discriminator(generated_y, training=False)
                        val_sum_acc += tf.reduce_sum(tf.round(pred_y)).numpy()
                        val_tot_acc += pred_y.shape[0]

                    # NOTE: validation loss only considers network performance to ensure early stopping is based
                    #       on our dominate goal
                    losses = tf.reduce_mean(loss_func(val_y, generated_y))
                    val_accum_loss += losses.numpy()
                    mae_i.update_state(val_y[:,:,:,0:1], generated_y[:,:,:,0:1])
                    mae_d.update_state(val_y[:,:,:,1:2], generated_y[:,:,:,1:2])
                    rmse_i.update_state(val_y[:,:,:,0:1], generated_y[:,:,:,0:1])
                    rmse_d.update_state(val_y[:,:,:,1:2], generated_y[:,:,:,1:2])

                    # add examples images to the tensorboard
                    for ii in range(val_y.shape[-1]):
                        board_callback(val_x[:,:,:,ii:ii+1], val_y[:,:,:,ii:ii+1], generated_y[:,:,:,ii:ii+1],
                                       step, epoch, f'{ii}/')

                val_accum_loss = tf.reduce_sum(val_accum_loss)
                tf.summary.scalar('loss: mae_i', mae_i.result().numpy() * 256., step=epoch)
                tf.summary.scalar('loss: mae_d', mae_d.result().numpy() * 2., step=epoch)
                tf.summary.scalar('loss: rmse_i', rmse_i.result().numpy() * 256., step=epoch)
                tf.summary.scalar('loss: rmse_d', rmse_d.result().numpy() * 2., step=epoch)
                tf.summary.scalar('loss: val', val_accum_loss, step=epoch)
                if discriminator is not None:
                    tf.summary.scalar('accuracy: val', val_sum_acc / val_tot_acc, step=epoch)

                # save the best model
                if val_accum_loss < global_loss:
                    print(f'Saving Network [{tf.keras.backend.get_value(generator.uuid).decode("utf-8")}]')
                    global_loss = val_accum_loss
                    tf.keras.models.save_model(generator, options.model_dir)
                    if discriminator is not None:
                        tf.keras.models.save_model(discriminator, os.path.join(options.model_dir, 'discriminator'))
                    num_worse = 0
                else:
                    num_worse += 1
                    print(f'Model did not achieve better results [{num_worse}]')
                    if num_worse > options.patience:
                        # reduce the learning rate for a few plateau events
                        if plateau_cnt < reduce_on_plateau:
                            plateau_cnt += 1
                            tf.keras.backend.set_value(generator.optimizer.learning_rate,
                                                       generator.optimizer.learning_rate / 5.)
                            num_worse = 0
                        else:
                            return
