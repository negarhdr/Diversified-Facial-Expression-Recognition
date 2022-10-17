# Learning Diversified Feature Representation for Facial Expression Recognition in the Wild

This repository contains the implementation of diversified feature representation learning method for facial expression recognition in the wild. 
This method is applied on two of the state-of-the-arts, ESR [[1]](#1) and MA-Net [[2]](#2), to improve their classification accuracy on three benchmark in-the-wild datasets. 
The codes are based on [ESR](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks) and [MA-Net](https://github.com/zengqunzhao/MA-Net) Github repositories. 

## Datasets
### AffectNet
To train models on the AffectNet dataset, first download and organize the dataset into the following structure:

```
AffectNet/    
    Training_Labeled/
        0/
        1/
        ...
        n/
    Training_Unlabeled/
        0/
        1/
        ...
        n/
    Validation/
        0/
        1/
        ...
        n/
```
The folder 0/, 1/, ..., /n contains up to 500 images from the AffectNet after pre-processing. To pre-process the images and organize them into the above structure, call the method pre_process_affect_net(base_path_to_images, base_path_to_annotations, base_destination_path, set_index=1) from ./utils/udata.py. For preprocessing the training data, set the parameter set_index=1, and for validation set_index=1. The images will be cropped (to get the face only) and re-scaled to image_size x image_size pixels. Following the experimental setting of ESR and MA-Net methods, the image_size for ESR is set to 96 and for MA-Net is set to 224. Each image is renamed to follow the pattern "[id]_[emotion_idx]_[valence times 1000]_[arousal times 1000].jpg".
After that, set the experimental variables including the base path to the dataset (base_path_to_dataset = "[...]/AffectNet/").
The AffectNet dataset is available [here](http://mohammadmahoor.com/affectnet/).

### FER+
To finetune the models on FER+ dataset, you need to download the FER 2013 dataset from [Kaggle/FERPlus](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and organize it into the following structure:

```
FER_2013/
    Dataset/
        Images/
            FER2013Train/
            FER2013Valid/
            FER2013Test/
        Labels/
            FER2013Train/
            FER2013Valid/
            FER2013Test/
```
After that, set the experimental variables including the base path to the dataset (base_path_to_dataset = "[...]/FER_2013/Dataset/").
Our experiments used the FER+ labels which are available at [Microsoft/FERPlus](https://github.com/microsoft/FERPlus).


### RAF-DB
To finetune the models on RAF-DB dataset, you need to download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/raf/model1.html), and make sure that it has the structure like following:
 
```
./RAF-DB/
         train/
               0/
                 train_09748.jpg
                 ...
                 train_12271.jpg
               1/
               ...
               6/
         test/
              0/
              ...
              6/

[Note] 0: Neutral; 1: Happiness; 2: Sadness; 3: Surprise; 4: Fear; 5: Disgust; 6: Anger
```


## Training 
For training the ESR method on AffectNet dataset, run ```esr_affectnet.py``` and specify the number of branches in the script as "num_branches_trained_network=9". 
The trained ESR model on AffectNet can be finetuned on FER+ and RAF-DB datasets by running ```esr_ferplus.py``` and ```esr_rafdb.py```, respectively. 

According to the MA-Net paper, it is pretrained on MS-Celeb-1M dataset the finetuned on facial expression recognition datasets. The pretrained model on MS-Celeb-1M, called ```Pretrained_on_MSCeleb.pth.tar``` can be downloaded from [here](https://drive.google.com/file/d/1tro_RCovLKNACt4MKYp3dmIvvxiOC2pi/view?usp=sharing) and put it in the ```checkpoint/MS-Celeb-1M/``` directory. 
This pretrained model can be finetuned on AffectNet, FER+ and RAF-DB by running ```manet_affectnet.py```, ```manet_ferplus.py```, and ```manet_rafdb.py```, respectively.
## Access to pre-trained models
To be updated soon 
## Citation
```
@article{heidari2022diversified,
  title={Learning diversified feature representations for facial expression recognition in the wild},
  author={Heidari, Negar and Iosifidis, Alexandros},
  journal={arXiv preprint arXiv:?},
  year={2022}
}
```

# Acknowledgements
<img src="https://github.com/negarhdr/Diversified-Facial-Expression-Recognition/blob/main/eu_logo.png" width="100"> This work received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

<img src="https://github.com/negarhdr/Diversified-Facial-Expression-Recognition/blob/main/opendr_logo.png" width="100"> 



# References 


<a id="1">[1]</a> 
[Siqueira, Henrique, Sven Magg, and Stefan Wermter. "Efficient facial feature learning with wide ensemble-based convolutional neural networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.](https://ojs.aaai.org/index.php/AAAI/article/view/6037)

<a id="2">[2]</a> 
[Zhao, Zengqun, Qingshan Liu, and Shanmin Wang. "Learning deep global multi-scale and local attention features for facial expression recognition in the wild." IEEE Transactions on Image Processing 30 (2021): 6544-6556.](https://ieeexplore.ieee.org/abstract/document/9474949?casa_token=EcE55deTQhIAAAAA:fGEO1hcE3J80KxgtGLwXPgpsGD-5maFFvddoMG3mim0RU9j1mR_jVuFjmPDGTXPcWWeuh8U)

<a id="3">[3]</a> 
[Mollahosseini, Ali, Behzad Hasani, and Mohammad H. Mahoor. "Affectnet: A database for facial expression, valence, and arousal computing in the wild." IEEE Transactions on Affective Computing 10.1 (2017): 18-31.](https://ieeexplore.ieee.org/abstract/document/8013713?casa_token=jNjpOPFaoGAAAAAA:_sI3UC3TdaFj2JHMZfrvlVev-DIwWHCOekhgF-IZ1I-nklm8DT1-KoW7kutALHbRLOQiPac)

<a id="4">[4]</a> 
[Barsoum, Emad, et al. "Training deep networks for facial expression recognition with crowd-sourced label distribution." Proceedings of the 18th ACM International Conference on Multimodal Interaction. 2016.](https://dl.acm.org/doi/abs/10.1145/2993148.2993165?casa_token=TKDVV7lRdP8AAAAA:Oik4YYGDt-L-_TBSUZFHfv2buvXkFLqlxqv3qXBFyutJk9Gsrdw3o2DSCQG5gunJ9w7QKB_fQg)

<a id="5">[5]</a> 
[Li, Shan, Weihong Deng, and JunPing Du. "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.html)
