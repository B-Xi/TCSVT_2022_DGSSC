DGSSC: A Deep Generative Spectral-Spatial Classifier for Imbalanced Hyperspectral Imagery, TCSVT, 2022
==
[Bobo Xi](https://scholar.google.com/citations?user=O4O-s4AAAAAJ&hl=zh-CN), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), Yan Diao, [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), [Zan Li](https://web.xidian.edu.cn/zanli/), [Yan Huang](https://radio.seu.edu.cn/2019/0301/c19950a264233/pagem.htm), and [Jocelyn Chanussot](https://jocelyn-chanussot.net/).
***
Code for the paper: [DGSSC: A Deep Generative Spectral-Spatial Classifier for Imbalanced Hyperspectral Imagery](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9924229).

<div align=center><img src="/image/frameworks.jpg" width="80%" height="80%"></div>
Fig. 1: Architecture of the proposed DGSSC for imbalanced hyperspectral imagery. The encoder, decoder, and classifier are trained in an end-to-end fashion,
driven by the integrated loss of L<sub>PdRec</sub>, L<sub>mmd</sub>, and L<sub>cls</sub>.

Training and Test Process
--
Please run the 'run.py' to reproduce the DGSSC results on [Lokia](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene) dataset. 

We have successfully tested it on Ubuntu 16.04 with Tensorflow 1.13.1. Below is the classification result when 1\% labeled samples are selected as the training set. 

<div align=center><p float="center">
<img src="/image/false_color.jpg" title="(a)"height="150" width="400"/>
<img src="/image/gt.jpg" height="150"width="400"/>  
<img src="/figure/hyrank/DGSSC_hyrank_7706_8531.png" height="150"width="400"/>
</p></div>
<div align=center>Fig. 2: The composite false-color image, groundtruth, and classification map of Lokia dataset.</div>  

References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, J. Li, Y. Diao, Y. Li, Z. Li, Y. Huang, and J. Chanussot, "DGSSC: A Deep Generative Spectral-Spatial Classifier for Imbalanced Hyperspectral Imagery," in IEEE Transactions on Circuits and Systems for Video Technology, pp. 1-1, 2022, doi: 10.1109/TCSVT.2022.3215513.

Citation Details
--
BibTeX entry:
```
@ARTICLE{Xi2022_TCSVT_DGSSC,
  author={Xi, Bobo and Li, Jiaojiao and Diao, Yan and Li, Yunsong and Li, Zan and Huang, Yan and Chanussot, Jocelyn},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DGSSC: A Deep Generative Spectral-Spatial Classifier for Imbalanced Hyperspectral Imagery}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3215513}}
```
 
Licensing
--
Copyright (C) 2022 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
