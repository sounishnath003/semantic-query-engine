# Open Domain Question Passage Retrival

Open-domain question answering systems heavily rely on efficient passage retrieval methods. This step helps in selecting the relevant candidate contexts for answering any question. Open-domain question answering systems usually follow a 2 step pipeline:

**Context Retriever:** Context retriever is responsible for getting in a small subset of passages that are relevant to the question and may contain answers.

**Machine Reader:** The machine reader is responsible for then identifying the correct answer from those sets of passages.

![TDS-Blog-Image](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*AM-U7qMOCXxiUP6UwQDTaQ.png)
Note: Both retriever-reader and retriever-generator open-domain (OD) QA stacks include an external data source (document store), retriever, and reader/generator.

## Dense Passage Retrieval

Dense Passage Retrieval (DPR) for ODQA was introduced in 2020 as an alternative to the traditional TF-IDF and BM25 techniques for passage retrieval.

### Usefulness

1. The paper that introduced DPR begins by stating that this new approach outperforms current Lucene (the document store) BM25 retrievers by a 9–19% passage retrieval accuracy [1].

2. DPR is able to outperform the traditional sparse retrieval methods for two key reasons:

3. Semantically similar words (“hey”, “hello”, “hey”) will not be viewed as a match by TF. DPR uses dense vectors encoded with semantic meaning (so “hey”, “hello”, and “hey” will closely match).
   Sparse retrievers are not trainable. DPR uses embedding functions that we can train and fine-tune for specific tasks.

## Two BERTs, And Training

DPR works by using **two unique BERT encoder** models.

\*\* One of those models — `[(Eᴘ)]` — encodes passages of text into an encoded passage vector (we store context vectors in our document store).

\*\* The other model — `[(EQ)]` — maps a question into an encoded question vector.

![TDS-Blog-Image](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*QoEy0MsJy2Wdl0S3GmqRKw.png)
Note: During training, we feed a question-context pair into our DPR model, and the model weights will be optimized to maximize the dot product between two respective Eᴘ/EQ model outputs:

### DensePassageRetrivalDeepNeuralNetM Architecture

```
INFO:root:DensePassageRetrivalDeepNeuralNetM(
  (passage_retrival_ranker): PassageContextRetrivalEncoder(
    (pretrained-bert-finetuned): HuggingFaceBertPyTorch(...),
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (query_reader_ranker): QueryContextReaderEncoder(
    (pretrained-bert-finetuned): HuggingFaceBertPyTorch(...),
    (dropout): Dropout(p=0.1, inplace=False)
  )
)
```

### Kudos to the links I followed

- [Bert WordPiece Tokenizer](https://towardsdatascience.com/how-to-build-a-wordpiece-tokenizer-for-bert-f505d97dddbb)
- [Understanding DPR Systems](https://towardsdatascience.com/understanding-dense-passage-retrieval-dpr-system-bce5aee4fd40)
- [Humanoid Question Answering Model Architecture](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00341-6)
- [Dense Passage Retrival Research Paper](https://arxiv.org/pdf/2004.04906.pdf)
- [Haystack Deepset Explanation On DPR Models](https://haystack.deepset.ai/tutorials/09_dpr_training)
- [Dense Passage Retrival Ranker Towards Data science Paper Explanation](https://towardsdatascience.com/how-to-create-an-answer-from-a-question-with-dpr-d76e29cc5d60)
