import tensorflow as tf


def create_mag_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.001,
                                          lr_decay_step=200000,
                                          lr_decay_rate=0.77,
                                          momentum=0.99,
                                          lamb=15.0,
                                          batch_size=8,
                                          image_size=288,
                                          label_size=36,
                                          scope='mag',
                                          model_name='resnet_v1_50')
    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_metagrasp_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.001,
                                          lr_decay_step=200000,
                                          lr_decay_rate=0.77,
                                          momentum=0.99,
                                          lamb=120.0,
                                          batch_size=16,
                                          image_size=288,
                                          label_size=288,
                                          scope='metagrasp',
                                          model_name='resnet_v1_50')
    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams

