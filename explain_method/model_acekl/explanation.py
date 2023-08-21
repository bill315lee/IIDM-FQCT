"""
Explanation class, with visualization functions.
"""
from __future__ import unicode_literals
from io import open
import os
import os.path
import json
import string
import numpy as np

class Explanation(object):
    """Object returned by explainers."""
    def __init__(self, feature_names, feature_values, scaled_row):
        """Initializer.

        Args:
            domain_mapper: must inherit from DomainMapper class
            class_names: list of class names
        """

        self.exp_feature_names = feature_names
        self.feature_values = feature_values
        self.scaled_row = scaled_row
        self.local_exp = None
        self.intercept = None
        self.predict_score = None

    def as_list(self):
        """Returns the explanation as a list.

        Returns:
            list of tuples (representation, weight), where representation is
            given by name. Weight is a float.
        """
        return [(names[x[0]], x[1]) for x in self.local_exp]

    def as_map(self):
        """Returns the map of explanations.

        Returns:
            Map from label to list of tuples (feature_id, weight).
        """
        # print self.local_exp
        return self.local_exp


