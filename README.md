<p align="center"><img src="spoopy/docs/figs/logo.png"></p>

[![Build Status](https://travis-ci.org/bresan/PADify.svg?branch=master)](https://travis-ci.org/bresan/PADify)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/bresan/spoopy/graphs/commit-activity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PADify (pronounced as /ˈpædfaɪ/) is a project to perform presentation attack detection (PAD) on publicly available datasets by using different intrinsic image properties along with Convolutional Neural Networks.

## Presentation Attack Detection
<p align="center"><img width="50%"src="spoopy/docs/figs/examples.png"></p>
  <p align="center">
<sub>Fig 1 - Examples of Presentation Attack Detection</sub>



## Pipeline Overview
<p align="center"><img src="spoopy/docs/figs/pipeline.png"></p>
  <p align="center">
<sub>Fig 2 - Overview of proposed method</sub>


## Properties

- Illuminant Maps ([Carvalho et al.](https://ieeexplore.ieee.org/document/6522874/))
- Saliency ([Zhu et al.](https://ieeexplore.ieee.org/document/6909756))
- Depth ([Godard et al.](https://arxiv.org/abs/1609.03677))

# Datasets

- [CASIA Face Anti-Spoofing Database](http://www.cbsr.ia.ac.cn/english/Databases.asp)
- [Replay Attack](https://www.idiap.ch/dataset/replayattack)
- [NUAA Imposter Database](http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html)
- [ROSE-Youtu Face Liveness Detection Dataset
](http://rose1.ntu.edu.sg/datasets/faceLivenessDetection.asp) (In Progress)

# License

This file is licensed under the MIT license. You can check the [LICENSE file](https://github.com/bresan/PADify/LICENSE) for more information.