"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-02-11 01:22:29
"""

import logging
import pandas as pd

from densePassageRetrival import (
    DensePassageRetrivalConfiguration,
    DensePassageRetrivalModel,
)


def load_dataset():
    train_data = [
        {
            "query": "Who is the protaganist of Dune?",
            "context": 'Dune is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or "spice", a drug that extends life and enhances mental abilities. Melange is also necessary for space navigation, which requires a kind of multidimensional awareness and foresight that only the drug provides. As melange can only be produced on Arrakis, control of the planet is a coveted and dangerous undertaking.',
        },
        {
            "query": "Who is the author of Dune?",
            "context": "Dune is a 1965 science fiction novel by American author Frank Herbert, originally published as two separate serials in Analog magazine.",
        },
    ]
    return pd.DataFrame(train_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dfx = load_dataset()
    logging.info(dfx)

    configuration = DensePassageRetrivalConfiguration()
    logging.info(configuration)

    dpr_model = DensePassageRetrivalModel(config=configuration)
    logging.info(dpr_model)
