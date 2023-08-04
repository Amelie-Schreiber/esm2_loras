# esm2_loras

This is an attempt at training a Low Rank Adaptation (LoRA) for the protein language model ESM-2 for a token classification task. In particular, we attempt to train an RNA binding site predictor. There are still some issues to work out and any feedback or advice would be much appreciated. 

To set up the the conda environment, clone the repo and run:
```
conda env create -f environment.yml
```
Then run:
```
conda activate lora_esm_2
```
