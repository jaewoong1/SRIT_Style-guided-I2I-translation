# Official Pytorch Implementation - SRIT (AAAI 2022)

> Official code for the paper : [Style-Guided and Disentangled Representation for Robust Image-to-Image translation](https://ojs.aaai.org/index.php/AAAI/article/view/19924)

Appendix can be found here [[Appedix]](https://drive.google.com/file/d/1Vl-MBJEe3bsSWRGsM6O85AtPG2L0N5zs/view?usp=sharing)

Authors : [Jaewoong Choi](https://github.com/jaewoong1),  [Daeha Kim](https://github.com/kdhht2334), [Byung Cheol Song](https://scholar.google.com/citations?user=yo-cOtMAAAAJ&hl=ko&oi=sra)

![iccv_fig1](https://user-images.githubusercontent.com/54341727/133568248-d9d83417-cd6b-404d-a408-50254940c3c4.png)


## Abstract
Recently,  various  image-to-image  translation  (I2I)  methodshave  improved  mode  diversity  and  visual  quality  in  termsof  neural  networks  or  regularization  terms.  However,  con-ventional  I2I  methods  relies  on  a  static  decision  boundaryand  the  encoded  representations  in  those  methods  are  en-tangled with each other, so they often face with ‘mode col-lapse’ phenomenon. To mitigate mode collapse, 1) we designa so-called style-guided discriminator that guides an input im-age to the target image style based on the strategy of flexi-ble decision boundary. 2) Also, we make the encoded repre-sentations include independent domain attributes. Based ontwo  ideas,  this  paper  proposes  Style-Guided  and  Disentan-gled Representation for Robust Image-to-Image Translation(SRIT). SRIT showed outstanding FID by 8%, 22.8%, and 10.1%  for  CelebA-HQ,  AFHQ,  and  Yosemite  datasets,  re-spectively. The translated images of SRIT reflect the stylesof target domain successfully. This indicates that SRIT showsbetter mode diversity than previous works.


![decision](https://user-images.githubusercontent.com/54341727/144345313-60725a60-94b5-4f69-8a30-b6916a08e11a.png)


## Requirements
Install the dependencies:
```
bash
conda create -n SRIT python=3.6.7
conda activate SRIT
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
pip install tqdm
```


## Pretrained model
>Click to download [Wing](https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt?dl=0) : Put checkpoint file in the following location. `./expr/checkpoints/`

>Click to download [pretrained SRIT](https://drive.google.com/drive/folders/1r8bwHYlce-PsRggmLj4NWKuQP9fTIXhT?usp=sharing)


## Training dataset
Put dataset folder in the following location. `./data/`

>Click to download [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ) [1]

>Click to download [AFHQ](https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0) [2]

>Click to download [Yosemite](https://www.kaggle.com/balraj98/summer2winter-yosemite/download) [3]


## Run 
>I2I translation for AFHQ
```
python main.py --mode sample --num_domain 3 --w_hpf 0 --train_img_dir ./data/afhq/train --val_img_dir ./data/afhq/val
```

>I2I translation for CelebA-HQ
```
python main.py --mode sample --num_domain 2 --w_hpf 1 --train_img_dir ./data/CelebA-HQ/train --val_img_dir ./data/CelebA-HQ/val
```

>I2I translation for Yosemite
```
python main.py --mode sample --num_domain 2 --w_hpf 0 --train_img_dir ./data/Yosemite/train --val_img_dir ./data/Yosemite/val 
```

>If you want to run the code on different sized (ex: 512) images, add the following code
```
--img_size 512
```

## Reference
[1] Karras, Tero, et al. "Progressive growing of gans for improved quality, stability, and variation." arXiv preprint arXiv:1710.10196 (2017).

[2] Choi, Yunjey, et al. "Stargan v2: Diverse image synthesis for multiple domains." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[3] Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

