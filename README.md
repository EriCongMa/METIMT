# METIMT


Modal Contrastive Learning based End-to-End Text Image Machine Translation

The official repository for TASLP Journal paper: 

- **Cong Ma**, Xu Han, Linghui Wu, Yaping Zhang, Yang Zhao, Yu Zhou, and Chengqing Zong. **Modal Contrastive Learning based End-to-End Text Image Machine Translation**. In IEEE/ACM Transactions on Audio, Speech, and Language Processing (IEEE/ACM TASLP), vol. 32, pp. 2153-2165, 2024, doi: 10.1109/TASLP.2023.3324540. [IEEE Xplore](https://ieeexplore.ieee.org/document/10284997/).



## 1. Introduction

Text image machine translation (TIMT) aims at directly translating text in the source language embedded in images into the target language. Most existing systems follow the cascaded pipeline diagram from recognition to translation, which suffers from the problem of error propagation, parameter redundancy, and information reduction. The end-to-end model has the potential to alleviate these issues via bridging the recognition and translation models. However, the challenge is the data limitation and modality gap between text and image. In this paper, we propose a novel end-to-end model, namely Modal contrastive learning based End- to-end Text Image Machine Translation (METIMT), which alleviates these issues through end-to-end text image machine translation architecture and modal contrastive learning. Specifically, an image encoder is designed to encode images into the same feature space of corresponding text sentences, with the guidance of an intra-modal and inter-modal contrastive learning module. To further promote the research of text image machine translation, we have constructed one synthetic and two real-world datasets. Extensive experiments show that our lighter, faster model outperforms not only existing pipeline methods but also state-of-the-art end-to-end models on both synthetic and real-world evaluation sets.



<img src="./Figures/model.jpg" style="zoom:60%;" />



## 2. Usage

### 2.1 Requirements

- python==3.6.2
- pytorch == 1.3.1
- torchvision==0.4.2
- numpy==1.19.1
- lmdb==0.99
- PIL==7.2.0
- jieba==0.42.1
- nltk==3.5
- six==1.15.0
- natsort==7.0.1



### 2.2 Train the Model

```shell
bash ./train_model_guide.sh
```



### 2.3 Evaluate the Model

```shell
bash ./test_model_guide.sh
```



### 2.4 Datasets

We use the dataset released in [E2E_TIT_With_MT](https://github.com/EriCongMa/E2E_TIT_With_MT/tree/main).



## 3. Acknowledgement

The reference code of the provided methods are:

- [EriCongMa](https://github.com/EriCongMa)/[**E2E_TIT_With_MT**](https://github.com/EriCongMa/E2E_TIT_With_MT)
- [clovaai](https://github.com/clovaai)/**[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)**
- [OpenNMT](https://github.com/OpenNMT)/**[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)**
- [THUNLP-MT](https://github.com/THUNLP-MT)/**[THUMT](https://github.com/THUNLP-MT/THUMT)**


We thanks for all these researchers who have made their codes publicly available.



## 4. Citation

If you want to cite our paper, please use this bibtex version:

- IEEEXplore offered bib citation format

  - ```latex
    @ARTICLE{10284997,
      author={Ma, Cong and Han, Xu and Wu, Linghui and Zhang, Yaping and Zhao, Yang and Zhou, Yu and Zong, Chengqing},
      journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
      title={Modal Contrastive Learning Based End-to-End Text Image Machine Translation}, 
      year={2024},
      volume={32},
      number={},
      pages={2153-2165},
      keywords={Transformers;Machine translation;Decoding;Semantics;Pipelines;Text recognition;Task analysis;Text image machine translation;contrastive learning;text image recognition;machine translation},
      doi={10.1109/TASLP.2023.3324540}}
    
    ```
  
- Semantic Scholar offered bib citation format

  - ```latex
    @article{Ma2024ModalCL,
      title={Modal Contrastive Learning Based End-to-End Text Image Machine Translation},
      author={Cong Ma and Xu Han and Linghui Wu and Yaping Zhang and Yang Zhao and Yu Zhou and Chengqing Zong},
      journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
      year={2024},
      volume={32},
      pages={2153-2165},
      url={https://api.semanticscholar.org/CorpusID:264107217}
    }
    ```

- DBLP offered bib citation format

  - ```latex
    @article{DBLP:journals/taslp/MaHWZZZZ24,
      author       = {Cong Ma and
                      Xu Han and
                      Linghui Wu and
                      Yaping Zhang and
                      Yang Zhao and
                      Yu Zhou and
                      Chengqing Zong},
      title        = {Modal Contrastive Learning Based End-to-End Text Image Machine Translation},
      journal      = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
      volume       = {32},
      pages        = {2153--2165},
      year         = {2024},
      url          = {https://doi.org/10.1109/TASLP.2023.3324540},
      doi          = {10.1109/TASLP.2023.3324540},
      timestamp    = {Fri, 16 Aug 2024 07:47:08 +0200},
      biburl       = {https://dblp.org/rec/journals/taslp/MaHWZZZZ24.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
    ```



If you have any issues, please contact with [email](macong275262544@outlook.com).
