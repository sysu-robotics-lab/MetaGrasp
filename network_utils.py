import tensorflow as tf
import tensorflow.contrib.slim as slim

import os


def decode_depth(encoded_depth):
    encoded_depth = tf.cast(encoded_depth, tf.float32)
    r, g, b = tf.unstack(encoded_depth, axis=2)
    depth = r * 65536.0 + g * 256.0 + b  # decode depth image
    # depth = tf.div(depth, tf.constant(100.0))
    return depth


def encode_depth(depth):
    # depth = tf.multiply(depth, tf.constant(10000.0))
    depth = tf.cast(depth, tf.uint16)
    r = depth / 256 / 256
    g = depth / 256
    b = depth % 256
    encoded_depth = tf.stack([r, g, b], axis=2)
    encoded_depth = tf.cast(encoded_depth, tf.uint8)
    return encoded_depth


def get_dataset(dataset_dir,
                num_readers,
                num_preprocessing_threads,
                image_size,
                label_size,
                batch_size=1,
                reader=None,
                shuffle=True,
                num_epochs=None,
                is_training=True,
                is_depth=True):
    dataset_dir_list = [os.path.join(dataset_dir, filename)
                        for filename in os.listdir(dataset_dir) if filename.endswith('.tfrecord')]
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/color': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded_depth': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/label': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/camera_height': tf.FixedLenFeature((), tf.float32, default_value=0.0),
    }
    items_to_handlers = {
        'color': slim.tfexample_decoder.Image(image_key='image/color',
                                              shape=(image_size, image_size, 3),
                                              channels=3),
        'encoded_depth': slim.tfexample_decoder.Image(image_key='image/encoded_depth',
                                                      shape=(image_size, image_size, 3),
                                                      channels=3),
        'label': slim.tfexample_decoder.Image(image_key='image/label',
                                              shape=(label_size, label_size, 3),
                                              channels=3),
        'camera_height': slim.tfexample_decoder.Tensor(tensor_key='image/camera_height',
                                                       shape=(1,)),

    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=dataset_dir_list,
                                   reader=reader,
                                   decoder=decoder,
                                   num_samples=3,
                                   items_to_descriptions=None)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=num_readers,
                                                              shuffle=shuffle,
                                                              num_epochs=num_epochs,
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size)
    color, encoded_depth, label, camera_height = provider.get(['color', 'encoded_depth', 'label', 'camera_height'])
    color = tf.cast(color, tf.float32)
    color = color * tf.random_normal(color.get_shape(), mean=1, stddev=0.01)
    color = color / tf.constant([255.0])
    color = (color - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    depth = decode_depth(encoded_depth)
    camera_height = tf.expand_dims(tf.expand_dims(camera_height, axis=0), axis=0)
    camera_height = camera_height / 1000.0
    depth = depth * tf.random_normal(depth.get_shape(), mean=1, stddev=0.01)
    depth = depth / 1000.0  # (depth - tf.reduce_mean(depth)) / 1000.0
    if is_depth:
        input = tf.concat([color, tf.expand_dims(depth, axis=2)], axis=2)
    else:
        input = color
    label = tf.cast(label, tf.float32)
    label = label / 255.0
    if is_training:
        inputs, labels, camera_heights = tf.train.batch([input, label, camera_height],
                                                        batch_size=batch_size,
                                                        num_threads=num_preprocessing_threads,
                                                        capacity=5*batch_size)
    else:
        inputs = tf.expand_dims(input, axis=0)
        labels = tf.expand_dims(label, axis=0)
        camera_heights = tf.expand_dims(camera_height, axis=0)
    return inputs, labels, camera_heights


def get_color_dataset(dataset_dir,
                      num_readers,
                      num_preprocessing_threads,
                      image_size,
                      label_size,
                      batch_size=1,
                      reader=None,
                      shuffle=True,
                      num_epochs=None,
                      is_training=True):
    dataset_dir_list = [os.path.join(dataset_dir, filename)
                        for filename in os.listdir(dataset_dir) if filename.endswith('.tfrecord')]
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/color': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/label': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    items_to_handlers = {
        'color': slim.tfexample_decoder.Image(image_key='image/color',
                                              shape=(image_size, image_size, 3),
                                              channels=3),
        'label': slim.tfexample_decoder.Image(image_key='image/label',
                                              shape=(label_size, label_size, 3),
                                              channels=3),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=dataset_dir_list,
                                   reader=reader,
                                   decoder=decoder,
                                   num_samples=3,
                                   items_to_descriptions=None)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=num_readers,
                                                              shuffle=shuffle,
                                                              num_epochs=num_epochs,
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size)
    color, label = provider.get(['color', 'label'])
    color = tf.cast(color, tf.float32)
    color = color * tf.random_normal(color.get_shape(), mean=1, stddev=0.01)
    color = color / tf.constant([255.0])
    color = (color - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    label = tf.cast(label, tf.float32) / 255.0
    if is_training:
        colors, labels = tf.train.batch([color, label],
                                        batch_size=batch_size,
                                        num_threads=num_preprocessing_threads,
                                        capacity=5*batch_size)
    else:
        colors = tf.expand_dims(color, axis=0)
        labels = tf.expand_dims(label, axis=0)
    return colors, labels


def get_depth_dataset(dataset_dir,
                      num_readers,
                      num_preprocessing_threads,
                      image_size,
                      label_size,
                      batch_size=1,
                      reader=None,
                      shuffle=True,
                      num_epochs=None,
                      is_training=True):
    dataset_dir_list = [os.path.join(dataset_dir, filename)
                        for filename in os.listdir(dataset_dir) if filename.endswith('.tfrecord')]
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/encoded_depth': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/label': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    items_to_handlers = {
        'encoded_depth': slim.tfexample_decoder.Image(image_key='image/encoded_depth',
                                                      shape=(image_size, image_size, 3),
                                                      channels=3),
        'label': slim.tfexample_decoder.Image(image_key='image/label',
                                              shape=(label_size, label_size, 3),
                                              channels=3),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=dataset_dir_list,
                                   reader=reader,
                                   decoder=decoder,
                                   num_samples=3,
                                   items_to_descriptions=None)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=num_readers,
                                                              shuffle=shuffle,
                                                              num_epochs=num_epochs,
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size)
    encoded_depth, label = provider.get(['encoded_depth', 'label'])
    depth = decode_depth(encoded_depth)
    depth = (depth - tf.reduce_mean(depth)) / 1000.0
    depth = depth * tf.random_normal(depth.get_shape(), mean=1, stddev=0.01)
    depth = tf.stack([depth, depth, depth], axis=2)
    label = tf.cast(label, tf.float32) / 255.0
    if is_training:
        depths, labels = tf.train.batch([depth, label],
                                        batch_size=batch_size,
                                        num_threads=num_preprocessing_threads,
                                        capacity=5*batch_size)
    else:
        depths = tf.expand_dims(depth, axis=0)
        labels = tf.expand_dims(label, axis=0)
    return depths, labels


def add_summary(inputs, labels, end_points, loss, hparams):
    h_b = int(hparams.batch_size/2)
    tf.summary.scalar(hparams.scope+'_loss', loss)
    tf.summary.image(hparams.scope+'_inputs_g', inputs[0:h_b])
    tf.summary.image(hparams.scope+'_inputs_r', inputs[h_b:])
    tf.summary.image(hparams.scope+'_labels_g', labels[0:h_b])
    tf.summary.image(hparams.scope+'_labels_r', labels[h_b:])
    # for i in range(1, 3):
    #     for j in range(64):
    #         tf.summary.image(scope + '/conv{}' + '_{}'.format(i, j),
    #                          end_points[scope + '/conv{}'.format(i)][0:1, :, :, j:j + 1])
    # tf.summary.image(scope + '/conv3', end_points[scope + '/conv3'])
    net = end_points['logits']
    infer_map = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    tf.summary.image(hparams.scope+'_inference_map_g', infer_map[0:h_b])
    tf.summary.image(hparams.scope+'_inference_map_r', infer_map[h_b:])
    # variable_list = slim.get_model_variables()
    # for var in variable_list:
    #     tf.summary.histogram(var.name[:-2], var)


def restore_map():
    variable_list = slim.get_model_variables()
    variables_to_restore = {var.op.name: var for var in variable_list}
    return variables_to_restore


def restore_from_classification_checkpoint(scope, model_name, checkpoint_exclude_scopes):
    variable_list = slim.get_model_variables(os.path.join(scope, 'feature_extractor'))
    for checkpoint_exclude_scope in checkpoint_exclude_scopes:
        variable_list = [var for var in variable_list if checkpoint_exclude_scope not in var.op.name]
    variables_to_restore = {}
    for var in variable_list:
        if var.name.startswith(os.path.join(scope, 'feature_extractor')):
            var_name = var.op.name.replace(os.path.join(scope, 'feature_extractor'), model_name)
            variables_to_restore[var_name] = var
    return variables_to_restore
