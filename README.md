# **Emotionally Enhanced Talking Face Generation**

This code is PyTorch implementation of the paper: _Emotionally Enhanced Talking Face Generation_

|📑 Original Paper|📰 Project Page|🌀 Demo|⚡ Live Testing
|:-:|:-:|:-:|:-:|
[Paper](https://arxiv.org/abs/2303.11548) | [Project Page](https://midas.iiitd.edu.in/emo/) | [Demo Video](https://youtu.be/AoWDmrMlhQI) | [Interactive Demo](https://midas.iiitd.edu.in/emo/)

![Model](/images/model.png)

--------
**Disclaimer**
--------
All results from this open-source code or our [demo website](https://midas.iiitd.edu.in/emo/) should only be used for research/academic/personal purposes only. 

Prerequisites
-------------
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`.
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.

License and Citation
----------
Theis repository can only be used for personal/research/non-commercial purposes. Please cite the following paper if you use this repository:
```
@misc{goyal2023emotionally,
      title={Emotionally Enhanced Talking Face Generation}, 
      author={Sahil Goyal and Shagun Uppal and Sarthak Bhagat and Yi Yu and Yifang Yin and Rajiv Ratn Shah},
      year={2023},
      eprint={2303.11548},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
##### Training the model
Run: 
```bash
python train.py --data_root preprocessed_dataset/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint> --emotion_disc_path <path_to_emotion_disc_checkpoint>
```
You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.

Evaluation
----------
Please check the `evaluation/` folder for the instructions.

Acknowledgements
----------
The code structure is inspired by this [Wav2Lip](https://github.com/Rudrabha/Wav2Lip). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models.