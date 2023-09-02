This is the online repository of the OOPSLA 2023 paper titled "Two Birds with One Stone: Boosting Code Generation and Code Search via a Generative Adversarial Network". We have released the source code and dataset.
The doi of our artifact is: https://doi.org/10.5281/zenodo.7824776.

* Dataset: Our evaluation is based on the large-scale CodeSearchNet dataset. Use the following command to download and preprocess the data:
```bash
cd dataset
bash run.sh 
cd ..
```
* Dependencies:
```
pip install -r requirements.txt
```

[Optional] We have built the tree-sitter parser stored at `evaluator/CodeBLEU/parser/languages.so`. If it doesn't work for you, it can be rebuilt with the following command:
```bash
cd evaluator/CodeBLEU/parser
bash build.sh
```

* Training
```bash
bash sh/train.sh [python/java] [CodeT5/Natgen] [CodeBERT/GraphCodeBERT]
```

* Evaluation
Evaluate generator:
```bash
bash sh/evaluate.sh [python/java] [CodeT5/Natgen]
```
Evaluate discriminator:
We evaluate the discriminator by reusing the code from [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch) and [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch). According to the Evaluate section in the corresponding model's Readme, replace `model_name_or_path` with the respective discriminator that you want to evaluate.