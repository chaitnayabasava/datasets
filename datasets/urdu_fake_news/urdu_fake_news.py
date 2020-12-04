"""Urdu Fake News Dataset"""

from __future__ import absolute_import, division, print_function

import glob
import os

import datasets

_CITATION = "@article{MaazUrdufake2020,\n"
_CITATION += "  author = {Amjad, Maaz and Sidorov, Grigori and Zhila, Alisa and "
_CITATION += "  G’{o}mez-Adorno, Helena and Voronkov, Ilia  and Gelbukh, Alexander},\n"
_CITATION += "  title = {Bend the Truth: A Benchmark Dataset for Fake News "
_CITATION += "  Detection in Urdu and Its Evaluation},\n"
_CITATION += "  journal={Journal of Intelligent & Fuzzy Systems},\n"
_CITATION += "  volume={39},\n"
_CITATION += "  number={2},\n"
_CITATION += "  pages={2457-2469},\n"
_CITATION += "  doi = {10.3233/JIFS-179905},\n"
_CITATION += "  year={2020},\n"
_CITATION += "  publisher={IOS Press}\n"
_CITATION += "}"

_DESCRIPTION = """
Urdu fake news datasets that contain news of 5 different news domains.
These domains are Sports, Health, Technology, Entertainment, and Business.
The real news are collected by combining manual approaches.
"""

_URL = "https://github.com/MaazAmjad/Datasets-for-Urdu-news/blob/master/"
_URL += "Urdu%20Fake%20News%20Dataset.zip?raw=true"

class UrduFakeNews(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    category_list = [
        "bus",
        "hlth",
        "sp",
        "tch",
        "sbz",
    ]

    def _info(self):
        labels_list = ["Fake", "Real"]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "news": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=labels_list),
                    "category": datasets.ClassLabel(names=self.category_list),
                }
            ),
            homepage="https://github.com/MaazAmjad/Datasets-for-Urdu-news",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download_and_extract(_URL)
        input_path = os.path.join(dl_path, "1.Corpus")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "pattern": os.path.join(input_path, "Train", "*", "*.txt")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "pattern": os.path.join(input_path, "Test", "*", "*.txt")
                },
            ),
        ]

    def _generate_examples(self, pattern=None):
        """Yields examples."""
        for filename in sorted(glob.glob(pattern)):

            with open(filename, encoding="utf-8") as f:
                news = ""
                for line in f:
                    if(line == '\n'):
                        continue
                    news += line

            name = os.path.basename(filename)
            key = name.rstrip(".txt")

            _class = 1 if("Real" in filename) else 0
            category = ''.join([i for i in key if not i.isdigit()])
            if(category != ""):
                category = self.category_list.index(category)
            else:
                category = 0

            yield key, {"news": news, "label": _class, "category": category}
