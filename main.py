from utils import extract_alloy, method_four
from joblib import load
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("attention_bi_lstm")
quantx = load("X_QuantileTransformer.bin")
quanty = load("Y_QuantileTransformer.bin")
ex_tree_reg = load("ExtraTreeRegressor.bin")


def get_inside(mod, layer):
    layer = mod.get_layer(layer)
    out = tf.keras.layers.Flatten()(layer.output)
    return tf.keras.Model(inputs=mod.input, outputs=out)


feature_model = get_inside(model, "dot")


def predict(alloys):
    elements, composition = extract_alloy(np.array(alloys))
    elemental_embeddings = method_four(elements, composition)
    dot_product_output = feature_model.predict(elemental_embeddings)
    transformed_dot = quantx.transform(dot_product_output)
    y_hat = ex_tree_reg.predict(transformed_dot)
    return quanty.inverse_transform(np.reshape(y_hat, (len(y_hat), 1)))


if __name__ == "__main__":
    print(predict(["Ti20.0Cu60.0Hf20.0"]))
