# utils.py
#
# The purpose of each function has been mentioned as per their occurrence.

import numpy as np
import pandas as pd
import re
from PyAstronomy import pyasl
import tensorflow as tf


# weights function extracts the elemental concentration from string form of alloy
def weights(string):
    weight = re.findall(r"[-+]?(?:\d*\.*\d+)", string)
    if len(weight) == 0:
        return float(1)
    return float(weight[0])


weights = np.vectorize(weights)


# names function extracts element names from string form of alloy
def names(string):
    return re.split(r"[-+]?(?:\d*\.*\d+)", string)[0]


names = np.vectorize(names)


# to_mode_fraction function converts elemental concentration to the form of mole fraction
def to_mole_fraction(array):
    total = np.sum(array)
    return array / total


# extract_alloy function take input a list of string form of alloy
# It extracts the element names and elemental concentration from each alloy
def extract_alloy(alloys):
    elements = []
    composition = []
    for i in range(len(alloys)):
        el_string = alloys[i]
        if '(' not in el_string and '[' not in el_string and '{' not in el_string:
            el_arr = np.array(re.findall('[A-Z][^A-Z]*', el_string))
            el_name = names(el_arr)
            el_con = weights(el_arr)
            if 0.0 not in to_mole_fraction(el_con):
                elements.append(el_name)
                composition.append(to_mole_fraction(el_con))
            else:
                return f"total not zero: {el_string}", "error"
        else:
            return f"have bracket: {el_string}", "error"
    return elements, composition


# Following are the different vectorization methods that have been employed

# Method - 1
#
# The alloys are represented a form of one-hot encoded, where zeros represent the absence of an element in the alloy
# and some value represents the concentration of that element
unique_elements = np.array(
    ['Ag', 'Al', 'Au', 'B', 'Be', 'Bi', 'C', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Fe', 'Ga', 'Gd',
     'Ge', 'Hf', 'Ho', 'In', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr',
     'Pt', 'Sc', 'Si', 'Sm', 'Sn', 'Ta', 'Tb', 'Ti', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'])


def method_one(elements, composition):
    vector = np.zeros((len(elements), len(unique_elements)))
    for i in range(len(elements)):
        for j in range(len(elements[i])):
            vector[i, np.where(unique_elements == elements[i][j])[0][0]] = composition[i][j]
    return vector


# Method - 2
#
# The alloys are represented using atomic numbers immediately followed by their respective concentration

MAX_LEN = 9


def element_to_index(element):
    """
    Parameters
    ----------
    element : str
        The element that you want the atomic number of
        Example: "Al"

    Returns
    -------
    atomic_number : int
        Example: 13
    """
    try:
        atomic_number = pyasl.AtomicNo()
        return atomic_number.getAtomicNo(element)
    except:
        return "END"


def method_two(elements, composition):
    vector = np.zeros((len(elements), MAX_LEN * 2))
    for i in range(len(elements)):
        for j in range(len(elements[i])):
            vector[i, j * 2] = element_to_index(elements[i][j])
            vector[i, j * 2 + 1] = composition[i][j]
    return vector


# Method - 3
#
# The alloys are represented using precomputed weights given in "elements_wvmodel_2016_12-15-20.csv"
# along with that the concentration of each element is considered as index for one-hot encoding on the same vector

element_weights = pd.read_csv("elements_wvmodel_2016_12-15-20.csv")


def method_three(elements, composition):
    params = []
    for i in range(len(elements)):
        x = np.zeros((MAX_LEN, 200))
        for j in range(len(elements[i])):
            x[j, :] = element_weights[elements[i][j]]
            x[j, int(composition[i][j] * 100)] = 1
        params.append(x)
    return np.array(params)


# Method - 4
#
# The alloys are represented using precomputed weights given in "elements_wvmodel_2016_12-15-20.csv" and the vector of
# each element is multiplied by its mole fraction concentration in the alloy
# This method showed most promising results

def method_four(elements, composition):
    params = []
    for i in range(len(elements)):
        x = np.zeros((MAX_LEN, 200))
        for j in range(len(elements[i])):
            x[j, :] = element_weights[elements[i][j]]
            x[j, :] = x[j, :] * composition[i][j]
        params.append(x)
    return np.array(params)


def method_five(elements, composition):
    vector = np.zeros((len(elements), MAX_LEN * 2), dtype=float)
    for i in range(len(elements)):
        for j in range(len(elements[i])):
            vector[i, j] = np.where(unique_elements == elements[i][j])[0][0] + 1
            vector[i, j + MAX_LEN] = composition[i][j]
    return vector


# LSTM_Vectorizer
#
# This function converts the alloys vector computed using method_four into a 128 dimension vector using a pretrained
# lstm network "model-9"

def LSTM_Vectorizer(model, params, layer_name):
    tf.random.set_seed(42)
    lstm = tf.keras.models.load_model(model)
    selected_layer = lstm.get_layer(layer_name)
    feature_model = tf.keras.Model(inputs=lstm.input, outputs=selected_layer.output)
    features = feature_model.predict(params)
    return features


if __name__ == "__main__":
    vec = method_five([['Ag', 'Cu', 'Zr']], [[0.076, 0.6, 0.33]])

    print(vec[0])
