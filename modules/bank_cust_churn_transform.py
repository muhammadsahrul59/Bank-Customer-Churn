"""Transform module for bank customer churn."""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "country": 3,
    "gender": 2,
}

NUMERICAL_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
]

LABEL_KEY = "churn"


def transformed_name(key):
    """Renaming transformed features.

    Args:
        key: The feature key to be transformed.

    Returns:
        The transformed feature name.
    """
    return key + "_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector.

    Args:
        label_tensor: Tensor representing the label (0 or 1).
        num_labels: Number of labels for the one-hot encoding.

    Returns:
        One-hot encoded label tensor.
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs: Map from feature keys to raw features.

    Returns:
        outputs: Map from feature keys to transformed features.
    """
    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(inputs[key], top_k=dim + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
