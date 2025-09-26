# RiboNN: A deep learning model to predict translation efficiency from mRNA sequence

This is a fork of original [RiboNN repo](https://github.com/Sanofi-Public/RiboNN) with the intention to modify the user interface to my needs. 

For more information, please refer to the [original repo](https://github.com/Sanofi-Public/RiboNN) and the [RiboNN paper](https://www.biorxiv.org/content/10.1101/2024.08.11.607362v2).

## Instalation

### System requirements

  This code has been tested on a system with 4 CPUs, 16 Gb RAM, and 1 NViDIA 10A GPU, with Ubuntu 20.04 as the OS (with CUDA Toolkit 11.3 installed). The required softwares are listed in environment.yml.

### Create conda environment

  Install mamba into "miniforge3/" in the home directory
  ```bash
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh -b 
  ~/miniforge3/bin/mamba shell init 
  source ~/.bashrc
  ```

  Download `environment.yml` file
  ```bash
  wget https://raw.githubusercontent.com/katarinagresova/RiboNN/refs/heads/main/environment.yml
  ```

  Create environment
  ```bash
  ~/miniforge3/bin/mamba env create -f environment.yaml
  ```

  Activate environment
  ```bash
  mamba activate RiboNN
  ```

### Install RiboNN package

  ```bash
  pip install git+https://github.com//katarinagresova/RiboNN.git
  ```

### Download pretrained weights

  ```
  mkdir models
  cd models
  wget https://zenodo.org/records/15375573/files/weights.zip
  unzip weights.zip
  mv weights/* .
  rm weights
  rm weights.zip
  ```

## Predictions

Input file:
  - a tab-separated text file 
  - with columns named "tx_id" (unique transcript IDs), "utr5_sequence", "cds_sequence" (including start and stop codons), and "utr3_sequence"
  - Alternatively, the file may have columns named "tx_id", "tx_sequence" (full transcript seuquences containing 5'UTR, CDS, and 3'UTR), "utr5_size" (lengths of the 5'UTRs), and "cds_size" (lengths of the CDSs). 

**Note:** Input transcripts with 5'UTRs longer than 1,381 nt or combined CDS and 3'UTR sizes larger than 11,937 nt will be excluded in the output. 


### From command line

  ```bash
  RiboNN-predict --input_file data/prediction_input1.txt --output_file "data/sample_output.tsv" --models_dir ./models
  ```
  
### From python

  ```python
  from RiboNN.predict import predict

  predict(
      input_file="data/prediction_input1.txt",
      output_file="data/sample_output.tsv",
      models_dir="./models",
  )
  ```

