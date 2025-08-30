# Speech DF Arena toolkit

### A simple tool to compute score file and metrics like EER, F1 and accuracy across SOTA speech deepfake detection model on any dataset 

With the growing advent of machine-generated speech, the scientific community is responding with valuable contributions to detect deepfakes. With research moving at such a rapid pace, it becomes challenging to keep track of generalizability of SOTA deepfake detection systems. This tool allows users to compute EER, accuracy and F1 scores on popular countermeasure systems on any dataset provided a standardized protocol format.

This tool accompanies the main leaderboard which  can be found on [Hugging face](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena)

Currently supported models:

- [AASIST](https://arxiv.org/abs/2110.01200)  
- [RawGatST](https://arxiv.org/abs/2107.12710)  
- [WavLM ECAPA](https://www.isca-archive.org/asvspoof_2024/kulkarni24_asvspoof.pdf)  
- [Hubert ECAPA](https://www.isca-archive.org/asvspoof_2024/kulkarni24_asvspoof.pdf)  
- [Wav2Vec2 ECAPA](https://www.isca-archive.org/asvspoof_2024/kulkarni24_asvspoof.pdf)  
- [TCM](https://arxiv.org/abs/2406.17376)  
- [Rawnet2](https://arxiv.org/pdf/2011.01108)  
- [XLSR+SLS](https://openreview.net/pdf?id=acJMIXJg2u)  
- [Wav2Vec2 AASIST](https://arxiv.org/pdf/2202.12233)  
- [Nes2NetX](https://arxiv.org/pdf/2504.05657)

# Usage 							
### 1. Preparation 

Before you proceed you need to set two environment variables
```
export DF_ARENA_CHECKPOINTS_DIR='path/to/checkpoint/dir'
export DF_ARENA_PROTOCOL_FILES_DIR='path/to/protocol/dir'
```
The checkpoints directory should be set to the path where model checkpoints for supported models reside. The checkpoint file should be named as per the files inside  `./Models`  i.e xlsr_sls.pt, tcm.pth etc. 
The protocol directory can be set to the directory where your protocol files reside.

NOTE- 
- XLSR SLS, TCM and Wav2Vec2-AASIST use the Wav2vec2 XLSR checkpoint which can be obtained from here and should be placed in the checkpoints directory with filename `xlsr2_300m.pt

- The configuration settings from the original config files for AASIST, RawNet2 and RawGatST have been hardcoded in the model definitions inside `./Models`
- Checkpoints and configuration files for some of the systems currently on the leaderboard can be found [here](https://drive.google.com/file/d/1iajJbXtrTDgyvxQYBA44V9_-nd9RaMzj/view?usp=sharing) 

## 2. Create metadata.csv for any desired dataset with below format:
```
file_name,label
/path/to/audio1,spoof
/path/to/audio2,bonafide
...

```
NOTE : The labels should contain "spoof" for spoofed samples and "bonafide" for real samples.
       All the file_name paths should be absolute 

### 3. Evaluation



Example usage : 
`$python evaluate.py [options]`

#### Options

- `--model_name <'all' or space seperated model names>` 
&nbsp;&nbsp;&nbsp; List of models to evaluate.Should be 'all' to evaluate all models from `./Models or space seperated list of models.
&nbsp;&nbsp;&nbsp; - &nbsp;&nbsp;&nbsp;Supported values = ['all', rawgat_st', 'wav2vec2_ecapa', 'hubert_ecapa', 'wav2vec2_aasist', 'aasist', 'tcm_add', 'rawnet_2', 'model_factory', 'xlsr_sls', 'wavlm_ecapa', 'nes2net_x']
- `--protocol_files <'all' or space seperated protocol names>` 
&nbsp;&nbsp;&nbsp;List of protocols to evaluate on. Should be 'all' to evaluate all models from `./protocol_files or space seperated list of desired protocols.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Supported values =  ['dfadd', 'add_2023_round_2', 'codecfake', 'asvspoof_2021_la', 'in_the_wild', 'asvspoof_2019', 'add_2022_track_1', 'fake_or_real', 'asvspoof_2024', 'add_2022_track_3', 'add_2023_round_1', 'librisevoc', 'asvspoof_2021_df', 'sonar']
- `--batch_size <int>`   Batch size to use 
- `--fix_length` Whether to trim audio to 4 seconds for evaluation. Used by almost all the open source models.
- `--num_workers <int>` Number of pytorch workers to use \

Running above cli generates `./logs` and `./scores` directories to store the logs and score files. Each run creates seperate directories based on the timestamp. Both average and pooled results are computed. Pooled results are computed based on the global threshold obtained on the datasets involded in that particular run for every model involved. Average results are computed by simply averaging the results obtained across datasets involded in that particular run every the model.

Score files for benchmarked systems can be found [here](https://drive.google.com/file/d/1pI-tvCZt4U__gGGLsCQMdZqLv_QBe4NW/view?usp=sharing)