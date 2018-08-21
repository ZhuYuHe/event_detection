import tensorflow as tf 

def get_config_proto(log_device_placement=True, allow_soft_placement=True):
    config_proto = tf.ConfigProto(
        log_device_placement = log_device_placement,
        allow_soft_placement = allow_soft_placement
    )
    return config_proto