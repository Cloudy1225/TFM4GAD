# TFM4GAD: Tabular Foundation Models are Strong Graph Anomaly Detectors

This repository contains the official implementation of the paper:

> **Tabular Foundation Models are Strong Graph Anomaly Detectors**  
> *Yunhui Liu, Tieke He, Yongchao Liu, Can Yi, Hong Jin, Chuntao Hong*  
> Nanjing University & Ant Group, China  

TFM4GAD is a simple yet effective framework that adapts **Tabular Foundation Models (TFMs)**, such as **LimiX** and **TabPFN**, for **Graph Anomaly Detection (GAD)**.  It bridges the structural gap between graphs and tabular data via *graph-to-table flattening*, enabling **in-context anomaly detection** without any graph-specific training.

---

## 📂 Project Structure

```
TabPFN
├─ckpts                           # Directory for pretrained model checkpoints (.ckpt)
└─tfm4gad                         # Core implementation of the TFM4GAD framework
        augment.py                # Graph-to-tabular feature augmentation (Laplacian, structural, neighborhood)
        dataloader.py             # Dataset loading and preprocessing for GADBench datasets
        icl.conf.yaml             # Configuration file for datasets, parameters, and in-context learning setup
        main.py                   # Main script to run experiments and inference
        misc.py                   # Utility functions (metrics, logging, random seeds, etc.)
```

Each backbone (`LimiX` and `TabPFN`) provides a self-contained implementation of the TFM4GAD framework with identical structure.  

---

## 🚀 Getting Started

### 1. Prepare Datasets

We use four public datasets from [GADBench](https://github.com/squareRoot3/GADBench): **Amazon**, **YelpChi**, **T-Finance**, and **T-Social**.  Download them from [Google Drive](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing) and unzip into `TFM4GAD/datasets`.

| Dataset   | Domain      | #Nodes    | #Edges     | #Dim. | Anomaly |
| :-------- | :---------- | :-------- | :--------- | :---- | :------ |
| Amazon    | Co-review   | 11,944    | 4,398,392  | 25    | 9.5%    |
| YelpChi   | Co-review   | 45,954    | 3,846,979  | 32    | 14.5%   |
| T-Finance | Transaction | 39,357    | 21,222,543 | 10    | 4.6%    |
| T-Social  | Social      | 5,781,065 | 73,105,508 | 10    | 3.0%    |

---

### 2. Prepare Tabular Foundation Models

#### 🧮 LimiX

1. Download the LimiX source code from [limix-ldm/LimiX](https://github.com/limix-ldm/LimiX) (we use commit [a50777d](https://github.com/limix-ldm/LimiX/tree/a50777da8d41607bc6ea467c7e16b4e051c1fb19));
2. Merge it with the directory `TFM4GAD/LimiX`;
3. Download pretrained checkpoints from Hugging Face:  
- [LimiX-16M.ckpt](https://huggingface.co/stableai-org/LimiX-16M/tree/main)  
- [LimiX-2M.ckpt](https://huggingface.co/stableai-org/LimiX-2M/tree/main)  
- and place them under `TFM4GAD/LimiX/ckpts`.

**License Notice:** Code under [Apache-2.0 License](https://github.com/limix-ldm/LimiX/blob/main/LICENSE.txt).  Weights are available for **academic research**, with commercial use requiring authorization.

---

#### 🧮 TabPFN

1. Install the library `pip install tabpfn`;
2. Download pretrained checkpoints from Hugging Face:

   * [tabpfn-v2-classifier.ckpt](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/tree/main)
   * [tabpfn-v2.5-classifier-v2.5_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_5/tree/main)
   * and place them under`TFM4GAD/TabPFN/ckpts`.


**License Notice:**

* **TabPFN-v2:** [Prior Labs License (Apache 2.0 + attribution)](https://priorlabs.ai/tabpfn-license/)
* **TabPFN-v2.5:** Non-commercial research license. For enterprise use, contact [sales@priorlabs.ai](mailto:sales@priorlabs.ai).

---

### 3. Run Experiments

All configurations are stored in `icl.conf.yaml`.

```bash
# Using LimiX backbone
cd TFM4GAD/LimiX/tfm4gad
python main.py --dataset amazon --model_ckpt "../ckpts/LimiX-16M.ckpt"

# Using TabPFN backbone
cd TFM4GAD/TabPFN/tfm4gad
python main.py --dataset amazon --model_ckpt "../ckpts/tabpfn-v2-classifier.ckpt"
```

---

## ✨ Highlights

* **One-for-All GAD:**
  No dataset-specific training — generalizes across graph domains.
* **Graph-to-Table Flattening:**
  Enriches node features with Laplacian embeddings, structural metrics (Degree, PageRank), and anomaly-sensitive neighborhood features.
* **In-Context Learning:**
  Leverages pretrained TFMs to infer anomaly patterns directly from a few labeled samples.
* **Strong Empirical Results:**
  Outperforms state-of-the-art graph anomaly detectors on Amazon, YelpChi, T-Finance, and T-Social, achieving up to +10% AUPRC gain over trained baselines.
* **No Fine-Tuning Needed:**
  Zero training or gradient updates — pure inference-based anomaly detection.

---

## 🙏 Acknowledgements

We thank the authors of [GADBench](https://github.com/squareRoot3/GADBench) for providing datasets and baselines that form the foundation of this work.

We also thank the developers of [LimiX](https://github.com/limix-ldm/LimiX) and [TabPFN](https://github.com/PriorLabs/TabPFN) for their remarkable contributions to tabular foundation modeling.

---

## 📜 License

* Code: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
* LimiX Weights: Subject to [LimiX Model License](https://github.com/limix-ldm/LimiX/blob/main/LICENSE.txt)
* TabPFN Weights: Subject to [Prior Labs License](https://priorlabs.ai/tabpfn-license/)
