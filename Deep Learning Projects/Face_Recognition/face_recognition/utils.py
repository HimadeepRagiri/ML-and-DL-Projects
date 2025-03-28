# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H_esV8-7ljy3xakQNa4KZgnILR0mLKaD
"""

import json
import os


def load_database(database_path="face_database/face_database.json"):
    if os.path.exists(database_path):
        with open(database_path, "r") as f:
            return json.load(f)
    return {}


def save_database(data, database_path="face_database/face_database.json"):
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    with open(database_path, "w") as f:
        json.dump(data, f, indent=4)