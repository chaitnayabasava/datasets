"""IMDB Urdu movie reviews dataset."""

from __future__ import absolute_import, division, print_function

import os
import csv

import datasets


_URL = {
    "train": "https://github.com/urduhack/resources/releases/download/imdb_urdu_reviews_v1.0.0/imdb_urdu_reviews_train.csv",
    "test" : "https://github.com/urduhack/resources/releases/download/imdb_urdu_reviews_v1.0.0/imdb_urdu_reviews_test.csv",
}

_DESCRIPTION = """
To increase the availability of sentiment analysis dataset for a low recourse
language like Urdu, the already available IMDB Dataset was
translated using google translator into Urdu language.
This is a binary classification dataset having two classes as positive and negative.
The reason behind using this dataset is high polarity for each class.
It contains 50k samples equally divided in two classes.
"""

_HOMEPAGE = "https://www.kaggle.com/akkefa/imdb-dataset-of-50k-movie-translated-urdu-reviews"

class Imdb(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "sentiment": datasets.ClassLabel(names=["positive", "negative"]),
                }
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download_and_extract(_URL)
        print(dl_path)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "filepath": os.path.join(dl_path, "imdb_urdu_reviews_train.csv")
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "filepath": os.path.join(dl_path, "imdb_urdu_reviews_test.csv")
                }
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            for id_, row in enumerate(reader):
                if id_ == 0:
                    continue
                print(row)
