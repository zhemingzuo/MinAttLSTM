# **MinAttLSTM: A Multi-scale Attention-driven Parallel LSTM Framework for Integrated Prediction of Mid-Winter Break-Ups on Canadian Rivers**
By [Zheming Zuo](https://scholar.google.co.uk/citations?user=jzpjf4UAAAAJ&hl=en)<sup>1</sup>, [Zoe Li](https://scholar.google.com/citations?user=q-NmQAEAAAAJ&hl=en)<sup>2</sup>, and [Shuo Wang](https://scholar.google.com/citations?user=kFBS6asAAAAJ&hl=en)<sup>1</sup><br/>
<sup>1</sup> 1School of Computer Science, University of Birmingham, Birmingham, B15 2TT, UK<br/>
<sup>2</sup> Department of Civil Engineering, McMaster University, Hamilton, ON L8S 4L8, Canada<br/>

## _Introduction_
This is an official implementation of our **Min**imal **Att**ention-driven **L**ong **S**hort-**T**erm **M**emory (**MinAttLSTM**).

MinAttLSTM is a high-efficiency, one-stage multi-task learning framework that leverages parallelised gating and cross-attention mechanisms to provide precise, real-time Mid-Winter Break-Ups (MWB) forecasting by seamlessly aligning seasonal priors with daily hydrodynamic sequences. Our work is partially inspired by _[Were RNNs All We Needed?](https://arxiv.org/pdf/2410.01201)_ and _[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)_.

Our manuscript is coming soon. Please stay tuned.

## _Contents_
1. [Preparation](#preparation)
2. [Installation](#installation)
3. [Run](#run)

### _Preparation_
Clone the github repository.
```Shell
  git https://github.com/zhemingzuo/MinAttLSTM --recursive
  cd MinAttLSTM
```

### _Installation_
Set up a conda environmentwith all dependencies as follows:
```Shell
  conda env create -f environment.yml
  source activate minattlstm_2026
```

### _Run_
Run our MinAttLSTM for MWB occurrence prediction via
```Shell
  python scripts/train_L1_MinAttLSTM.py
```

Run our MinAttLSTM for MWB timing prediction  via
```Shell
  python scripts/train_L2_MinAttLSTM.py
```

Run our MinAttLSTM for integrated MWB multi-task monitoring via
```Shell
  python scripts/train_OneStage_MinAttLSTM.py
```