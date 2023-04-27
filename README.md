# **Emotionally Enhanced Talking Face Generation**

[![GitHub Stars](https://img.shields.io/github/stars/sahilg06/EmoGen)](https://github.com/sahilg06/EmoGen)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=sahilg06/EmoGen)




https://user-images.githubusercontent.com/59660566/233766823-a9d55fbf-32e0-4953-a033-a01fcbd6140e.mp4



This repository is the official PyTorch implementation of our paper: _Emotionally Enhanced Talking Face Generation_. We introduce a multimodal framework to generate lipsynced videos agnostic to any arbitrary identity, language, and emotion. Our proposed framework is equipped with a user-friendly [web interface](https://midas.iiitd.edu.in/emo/) with a real-time experience for talking face generation with emotions.

![Model](/images/model.png)



|📑 Original Paper|📰 Project Page|🌀 Demo|⚡ Live Testing
|:-:|:-:|:-:|:-:|
[Paper](https://arxiv.org/abs/2303.11548) | [Project Page](https://midas.iiitd.edu.in/emo/) | [Demo Video](https://youtu.be/bYPX0zp4MY4) | [Interactive Demo](https://midas.iiitd.edu.in/emo/)

**Disclaimer**
--------
All results from this open-source code or our [demo website](https://midas.iiitd.edu.in/emo/) should only be used for research/academic/personal purposes only. 

Prerequisites
-------------
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`.
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.

Preparing CREMA-D for training
----------

#### Download data
Download the data from this [repo](https://github.com/CheyneyComputerScience/CREMA-D).

#### Convert videos to 25 fps
```bash
python convertFPS.py -i <raw_video_folder> -o <folder_to_save_25fps_videos>
```

#### Preprocess dataset
```bash
python preprocess_crema-d.py --data_root <folder_of_25fps_videos> --preprocessed_root preprocessed_dataset/
```

Train!
----------
There are three major steps: (i) Train the expert lip-sync discriminator, (ii) Train the emotion discriminator (iii) Train the EmoGen model.

#### Training the expert discriminator
```bash
python color_syncnet_train.py --data_root preprocessed_dataset/ --checkpoint_dir <folder_to_save_checkpoints>
```
#### Training the emotion discriminator
```bash
python emotion_disc_train.py -i preprocessed_dataset/ -o <folder_to_save_checkpoints>
```

#### Training the main model
```bash
python train.py --data_root preprocessed_dataset/ --checkpoint_dir <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint> --emotion_disc_path <path_to_emotion_disc_checkpoint>
```
You can also set additional less commonly-used hyper-parameters at the bottom of the `hparams.py` file.

Inference
-------
<!-- #### Model Weights
| Model description| Link to the model |
| :-------------: | :---------------: |
| Emogen (PL+DA) | [Link](https://drive.google.com/file/d/1yNytUV2qI9RRbB_NMPy-Hgo4b0a-d76F/view?usp=sharing) |
| Emogen (PRE) | [Link](https://drive.google.com/file/d/1Z_J4xJmlyjue8Th8bl95cC60kOD4sZlZ/view?usp=sharing) | -->

Comment these code lines for inference: [line1](https://github.com/sahilg06/EmoGen/blob/7e58e8343dc0faff2685302920750cb8f7227651/models/wav2lip.py#L108) and [line2](https://github.com/sahilg06/EmoGen/blob/7e58e8343dc0faff2685302920750cb8f7227651/models/wav2lip.py#L113). 

```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> --emotion <categorical emotion>
```
The result is saved (by default) in `results/{emotion}.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio. Choose categorical emotion from this list: [HAP, SAD, FEA, ANG, DIS, NEU].

#### Tips for better results:
- Experiment with the `--pads` argument to adjust the detected face bounding box. Often leads to improved results. You might need to increase the bottom padding to include the chin region. E.g. `--pads 0 20 0 0`.
- If you see the mouth position dislocated or some weird artifacts such as two mouths, then it can be because of over-smoothing the face detections. Use the `--nosmooth` argument and give another try. 
- Experiment with the `--resize_factor` argument, to get a lower resolution video. Why? The models are trained on faces which were at a lower resolution. You might get better, visually pleasing results for 720p videos than for 1080p videos (in many cases, the latter works well too). 

Evaluation
----------
Please check the `evaluation/` folder for the instructions.

License and Citation
----------
This repository can only be used for personal/research/non-commercial purposes. Please cite the following paper if you use this repository:
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

Acknowledgements
----------
The code structure is inspired by [Wav2Lip](https://github.com/Rudrabha/Wav2Lip). We thank the authors for the wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models. Demo website is developed by [@ddhroov10](https://github.com/ddhroov10) and [@SakshatMali](https://github.com/SakshatMali).
