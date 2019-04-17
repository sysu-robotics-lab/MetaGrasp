import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import resnet_v1


def arg_scope(weight_decay=0.0005,
              batch_norm_decay=0.997,
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              activation_fn=tf.nn.relu,
              use_batch_norm=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def resize_bilinear(inputs,
                    height,
                    width,
                    outputs_collections=None,
                    scope=None):
    with tf.variable_scope(scope, 'resize', [inputs]) as sc:
        outputs = tf.image.resize_bilinear(inputs, [height, width], name='resize_bilinear')
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, outputs)


# multi-affordance grasping
def mag(inputs,
        num_classes=3,
        num_channels=1000,
        is_training=True,
        global_pool=False,
        output_stride=16,
        upsample_ratio=2,
        spatial_squeeze=False,
        reuse=tf.AUTO_REUSE,
        scope='graspnet'):
    with tf.variable_scope(scope, 'graspnet', [inputs], reuse=reuse):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                     num_classes=num_channels,
                                                     is_training=is_training,
                                                     global_pool=global_pool,
                                                     output_stride=output_stride,
                                                     spatial_squeeze=spatial_squeeze,
                                                     scope='feature_extractor')
        with tf.variable_scope('prediction', [net]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # to do: add batch normalization to the following conv layers.
            with slim.arg_scope([slim.conv2d],
                                outputs_collections=end_points_collection):
                net = slim.conv2d(net, 512, [1, 1], scope='conv1')
                net = slim.conv2d(net, 128, [1, 1], scope='conv2')
                net = slim.conv2d(net, num_classes, [1, 1], scope='conv3')
                height, width = net.get_shape().as_list()[1:3]
                net = tf.image.resize_bilinear(net,
                                               [height * upsample_ratio, width * upsample_ratio],
                                               name='resize_bilinear')
                end_points.update(slim.utils.convert_collection_to_dict(end_points_collection))
    end_points['logits'] = net
    return net, end_points


def metagrasp(inputs,
              num_classes=3,
              num_channels=1000,
              is_training=True,
              global_pool=False,
              output_stride=16,
              spatial_squeeze=False,
              reuse=tf.AUTO_REUSE,
              scope='metagrasp'):
    with tf.variable_scope(scope, 'metagrasp', [inputs], reuse=reuse):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                     num_classes=num_channels,
                                                     is_training=is_training,
                                                     global_pool=global_pool,
                                                     output_stride=output_stride,
                                                     spatial_squeeze=spatial_squeeze,
                                                     scope='feature_extractor')
        with tf.variable_scope('prediction', [net]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # to do: add batch normalization to the following conv layers.
            with slim.arg_scope([slim.conv2d, resize_bilinear],
                                outputs_collections=end_points_collection):
                net = slim.conv2d(net, 512, [3, 3], scope='conv1')
                height, width = net.get_shape().as_list()[1:3]
                net = resize_bilinear(net, height * 2, width * 2, scope='resize_bilinear1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv2')
                height, width = net.get_shape().as_list()[1:3]
                net = resize_bilinear(net, height * 2, width * 2, scope='resize_bilinear2')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3')
                height, width = net.get_shape().as_list()[1:3]
                net = resize_bilinear(net, height * 2, width * 2, scope='resize_bilinear3')
                net = slim.conv2d(net, 64, [3, 3], scope='conv4')
                height, width = net.get_shape().as_list()[1:3]
                net = resize_bilinear(net, height * 2, width * 2, scope='resize_bilinear4')
                net = slim.conv2d(net, num_classes, [5, 5], scope='conv5')
                end_points.update(slim.utils.convert_collection_to_dict(end_points_collection))
    end_points['logits'] = net
    return net, end_points
