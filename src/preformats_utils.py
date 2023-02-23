"""
# _* coding: utf8 *_

filename: preformat_dataset.py

@author: sounishnath
createdAt: 2023-02-19 01:23:09
"""

import json

from tqdm import tqdm

from densePassageRetrivalModel.types import (
    QA,
    Answer,
    Datum,
    Paragraph,
    QuestionAnswerDatasetType,
    QuestionAnswerDocument,
)


class PreformatDatasetProcessor:
    @staticmethod
    def LoadDataset(filepath: str):
        def __load_dataset(dataset_path: str):
            __datas: list[QuestionAnswerDocument] = []
            for record_data in PreformatDatasetProcessor.LoadJSON(
                dataset_filepath=dataset_path
            ):
                __datas.extend(PreformatDatasetProcessor.ProcessData(record_data))
            return (d for d in __datas)

        return __load_dataset(dataset_path=filepath)

    @staticmethod
    def LoadJSON(dataset_filepath: str):
        rawdata: QuestionAnswerDatasetType = QuestionAnswerDatasetType(
            **json.load(open(dataset_filepath))
        )
        return (Datum(**rdata) for rdata in tqdm(rawdata.data, total=len(rawdata.data)))

    @staticmethod
    def ProcessData(data: Datum):
        # data.paragraphs[0].qas[0].answers[0]
        results: list[QuestionAnswerDocument] = []

        for _rparag in tqdm(data.paragraphs, total=len(data.paragraphs)):
            rparag = Paragraph(**_rparag)
            for _rqas in rparag.qas:
                rqas = QA(**_rqas)
                for _rans in rqas.answers:
                    ques_ddata = QuestionAnswerDocument(
                        context=rparag.context,
                        query=rqas.question,
                        answer=Answer(**_rans),
                    )
                    results.append(ques_ddata)

        return (d for d in results)
