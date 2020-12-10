# Part-dependent Label Noise 
(Code coming soon!)

NeurIPS‘20: Part-dependent Label Noise: Towards Instance-dependent Label Noise (PyTorch implementation).  

This is the code for the paper:
[Part-dependent Label Noise: Towards Instance-dependent Label Noise](https://arxiv.org/pdf/2006.07836.pdf)      
Xiaobo Xia, Tongliang Liu, Bo Han, Nannan Wang, Mingming Gong, Haifeng Liu, Gang Niu, Dacheng Tao, Masashi Sugiyama.


## Dependencies
We implement our methods by PyTorch on NVIDIA Tesla V100 GPU. The environment is as bellow:
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version = 1.2.0
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 10.0
- [Anaconda3](https://www.anaconda.com/)

### Install requirements.txt
~~~
pip install -r requirements.txt
~~~

## Experiments
We verify the effectiveness of the proposed method on synthetic noisy datasets. In this repository, we provide the used [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format). You should put the datasets in the folder “data” when you have downloaded them.       
Here is a training example: 
```bash
python main.py \
    --dataset mnist \
    --noise_rate 0.2 \
    --seed 1 \
    --gpu 0
```
If you find this code useful in your research, please cite  
```bash
@inproceedings{xia2020part,
  title={Part-dependent Label Noise: Towards Instance-dependent Label Noise},
  author={Xia, Xiaobo and Liu, Tongliang and Han, Bo and Wang, Nannan and Gong, Mingming and Liu, Haifeng and Niu, Gang and Tao, Dacheng and Sugiyama, Masashi},
  booktitle={NeurIPS},
  year={2020}
}
```  
