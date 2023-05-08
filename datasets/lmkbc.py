# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""REBEL"""

from __future__ import absolute_import, division, print_function

import pandas as pd

import datasets

import re 
import json
import logging
import math
from collections import defaultdict

_DESCRIPTION = """\
REBEL is a silver dataset created for the paper REBEL: Relation Extraction By End-to-end Language generation
"""

_URL = ""
_URLS = {
    "train": _URL + "en_train.jsonl",
    "dev": _URL + "en_val.jsonl",
    "test": _URL + "en_test.jsonl",
}


class RebelConfig(datasets.BuilderConfig):
    """BuilderConfig for REBEL."""

    def __init__(self, **kwargs):
        """BuilderConfig for REBEL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RebelConfig, self).__init__(**kwargs)


class Rebel(datasets.GeneratorBasedBuilder):
    """Rebel 1.0"""

    BUILDER_CONFIGS = [
        RebelConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="",
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"], #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"], #self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        #relations_df = pd.read_csv(self.config.data_files['relations'], header = None, sep='\t')
        #relations = list(relations_df[0])

        with open(filepath[0], encoding="utf-8") as f:
            for id_, row in enumerate(f):
                article = json.loads(row)
                article["id"] = id_
                yield id_, article
