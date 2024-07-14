### 📋 [DocDeshadower: Frequency-aware Transformer for Document Shadow Removal](https://arxiv.org/abs/2307.15318)

<div>
<span class="author-block">
  Ziyang Zhou<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    Yingtie Lei<sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    Xuhang Chen<sup>📮</sup>
  </span>,
  <span class="author-block">
    Shenghong Luo
  </span>,
  <span class="author-block">
    Wenjun Zhang
  </span>,
  <span class="author-block">
    Chi-Man Pun<sup>📮</sup>
  </span>,
  <span class="author-block">
  Zhen Wang
</span>
  (👨‍💻‍ Equal contributions, 📮 Corresponding Author)
  </div>

<b>Huizhou University, University of Macau, Tp-Link International Shenzhen Co.,Ltd.</b>

In <b>_IEEE International Conference on Systems, Man, and Cybernetics 2024 (SMC 2024)_<b>

## ⚙️ Usage

### Training
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TRAINING in traning.yml

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties on the usage of accelerate, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

### Inference
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in traning.yml
```
python test.py
```

# 💗 Acknowledgements
This work was supported in part by the Guangdong Provincial Key R&D Programme under Grant No.2023B1111050010 and No.2020B0101100001, in part by the Huizhou Daya Bay Science and Technology Planning Project under Grant No.2020020003.

### 🛎 Citation
If you find our work helpful for your research, please cite:
```bib
```
