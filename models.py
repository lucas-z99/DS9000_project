"""
One place to build any model, so training and evaluation always agree.
"""

from CNN_Classifier import CNN_Classifier
from NN_Classifier import NN_Classifier

MODEL_TYPES = ("mlp", "cnn")


def build_model(model_type, n_class, image_size, n1, n2, dropout_p=0):
    if model_type == "mlp":
        return NN_Classifier(
            n_class=n_class,
            n_input=image_size * image_size,
            n1=n1,
            n2=n2,
            dropout_p=dropout_p,
        )
    if model_type == "cnn":
        return CNN_Classifier(
            n_class=n_class,
            image_size=image_size,
            n1=n1,
            n2=n2,
            dropout_p=dropout_p,
        )
    raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type}")


def wants_flat_input(model_type):
    """MLP takes a flat vector, CNN takes the 2D image."""
    return model_type == "mlp"


def count_params(model):
    return sum(p.numel() for p in model.parameters())
