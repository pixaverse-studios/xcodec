# X-Codec-2.0
Paper: LLaSA: Scaling Train Time and Test Time Compute for LLaMA based Speech Synthesis (Comming Soon!)


## Setup
Code is tested on `python3.9`

Please follow the following steps to setup your environment
1. Clone this repo
2. conda create --name xcodec2 python=3.9 
3. conda activate xcodec2  
2. `pip install -r requirements.txt`
3. Download the pretrained checkpoint by
```bash
wget  
```

## Inference
```bash
python inference.py  
```
The above cmd reconstruct all `.wav` files under the input directory and write the results to the output directory using the checkpoint.

BigCodec extracts a single token to represent each frame of the utterance. Refer to `inference.py` to find how to get the code.

## Train
To train a XCodec2, firstly you have to prepare your data 

1. Make a file list by:
```bash
python get_tsv.py
```

2. Train a X-Codec-2.0 with the default setting by:

```bash
python train.py log_dir=/path/to/log_dir
```

## Large scaling training and code extracting:

Training
```bash
Sbatch train_slurm.sh
```

Code extracting
```bash
Sbatch large_scale_save_code.sh
```

Codec will save in output folder with the same subfolder structure for audio file.


 
