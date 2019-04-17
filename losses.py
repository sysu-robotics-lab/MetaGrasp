import tensorflow as tf


def create_loss(net, labels):
    y = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    cross_entropy = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(y, 0.001, 0.999)))
    return cross_entropy


def create_loss_with_label_mask(net, labels, lamb):
    bad, good, background = tf.unstack(labels, axis=3)
    mask = lamb * tf.add(bad, good) + background * 0.1
    attention_mask = tf.stack([mask, mask, mask], axis=3)
    y = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    cross_entropy = -tf.reduce_mean(attention_mask * (labels * tf.log(tf.clip_by_value(y, 0.001, 0.999))))
    return cross_entropy


def create_loss_without_background(net, labels):
    bad, good, background = tf.unstack(labels, axis=3)
    background = tf.zeros_like(background, dtype=tf.float32)
    labels = tf.stack([bad, good, background], axis=3)
    y = tf.exp(net) / tf.reduce_sum(tf.exp(net), axis=3, keepdims=True)
    cross_entropy = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(y, 0.001, 0.999)))
    return cross_entropy
