# Fine-tunnig BERT on Multilingual Binary Classification
---

├── bert-base-multilingual-uncased
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── config.py
├── data
│   ├── jigsaw-toxic-comment-train-processed-seqlen128.csv
│   ├── jigsaw-toxic-comment-train.csv
│   ├── jigsaw-unintended-bias-train-processed-seqlen128.csv
│   ├── jigsaw-unintended-bias-train.csv
│   ├── sample_submission.csv
│   ├── test-processed-seqlen128.csv
│   ├── test.csv
│   ├── validation-processed-seqlen128.csv
│   └── validation.csv
├── dataset.py
├── docker.sh
├── engine.py
├── model.py
├── pytorch-xla-env-setup.py
└── train.py