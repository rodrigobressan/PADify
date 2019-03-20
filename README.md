<p align="center"><img src="spoopy/docs/figs/logo.png"></p>


[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/bresan/spoopy/graphs/commit-activity)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PADify (pronounced as /ˈpædfaɪ/) is a project to perform presentation attack detection (PAD) on publicly available datasets by using different intrinsic image properties along with Convolutional Neural Networks.


## Pipeline Overview
<p align="center"><img src="spoopy/docs/figs/logo.png"></p>
  <p align="center">
<sub>Fig 1 - Overview of proposed method</sub>


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

## Results

### Intra-Dataset Evaluation

#### CASIA
<p align="center">
  <img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/cbsr_all_roc.png" width="500" />

  </p>

</p>
<p align="center" width="100%">
 <img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/cbsr_cut_roc.png" width="250" />
 <img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/cbsr_print_roc.png" width="250" />
<img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/cbsr_tablet_roc.png" width="250" />
 </p>


#### Replay Attack

  <p align="center">
  <img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/ra_all_roc.png" width="500" />

  </p>

</p>
<p align="center" width="100%">
 <img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/ra_highdef_roc.png" width="250" />
 <img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/ra_mobile_roc.png" width="250" />
<img src="https://raw.githubusercontent.com/bresan/spoopy/master/spoopy/tools/visualization/roc/ra_print_roc.png" width="250" />
 </p>

# Modules

- [Feature Vectors Classifier](https://github.com/bresan/spoopy/tree/master/spoopy/tools/classifier): classification on the extracted feature vectors with ResNet50. Currently using only SVM.
- [Probability Classifier](https://github.com/bresan/spoopy/tree/master/spoopy/tools/classifier_probas): classification on the probabilities generated from feature vectors classifier. Currently using SVM and XGBoosting.
- [Data Manipulation](https://github.com/bresan/spoopy/tree/master/spoopy/tools/data): perform manipulation on datasets (separate by attack type, quality), as well as check files integrity.
- [Face Aligner](https://github.com/bresan/spoopy/tree/master/spoopy/tools/face_aligner): face alignment on images using OpenCV and Dlib.
- [Face Detector](https://github.com/bresan/spoopy/tree/master/spoopy/tools/face_detector): face recognition using OpenCV.
- [Feature Extractor](https://github.com/bresan/spoopy/tree/master/spoopy/tools/feature_extractor): extract bottleneck features using ResNet50.
- [Property Extractor](https://github.com/bresan/spoopy/tree/master/spoopy/tools/map_extractor): wrapper to call the property extractors in a single place.
- [Saliency Extractor](https://github.com/bresan/spoopy/tree/master/spoopy/tools/saliency): generate saliency maps using the method proposed by Zhu et al.
- [Depth Extractor](https://github.com/bresan/spoopy/tree/master/spoopy/tools/depth): generate depth maps using the method proposed by Godard et al (monodepth).
- [Illuminant Extractor](https://github.com/bresan/spoopy/tree/master/spoopy/tools/vole): generate illuminant maps using the method proposed by Carvalho et al.
- [Visualization](https://github.com/bresan/spoopy/tree/master/spoopy/tools/visualization): creation of ROC curves and result tables.


# Requirements

- [Python 3.5.0](https://www.python.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCv](https://opencv.org/)
- [Matploblib](https://matplotlib.org/)
- [ImageIo](http://imageio.github.io/)
- [Dlib](http://dlib.net/)

# Board

You can check the project current issues and features through the projects section (on top, below the project name).

# License

This file is licensed under the MIT license. You can check the [LICENSE file](https://github.com/bresan/spooky/LICENSE) for more information.

# References

- G. Pan, Z. Wu, and L. Sun, “Liveness detection for face recognition,”
  in Recent Advances in Face Recognition, K. Delac, M. Grgic, and M. S.
  Bartlett, Eds. Rijeka: IntechOpen, 2008, ch. 9.
- A. d. S. Pinto, H. Pedrini, W. Schwartz, and A. Rocha, “Video-based
  face spoofing detection through visual rhythm analysis,” in 2012 25th
  SIBGRAPI Conference on Graphics, Patterns and Images, Aug 2012,
  pp. 221–228.
- J. Maatta, A. Hadid, and M. Pietikainen, “Face spoofing detection from ¨
 single images using micro-texture analysis,” in 2011 International Joint
 Conference on Biometrics (IJCB), Oct 2011, pp. 1–7.
- J. Komulainen, A. Hadid, and M. Pietikainen, “Context based face anti- ¨
  spoofing,” in 2013 IEEE Sixth International Conference on Biometrics:
  Theory, Applications and Systems (BTAS), Sept 2013, pp. 1–8.
- J. Yang, Z. Lei, S. Liao, and S. Z. Li, “Face liveness detection with
  component dependent descriptor,” in 2013 International Conference on
  Biometrics (ICB), June 2013, pp. 1–6
- B. Peixoto, C. Michelassi, and A. Rocha, “Face liveness detection
  under bad illumination conditions,” in 2011 18th IEEE International
  Conference on Image Processing, Sept 2011, pp. 3557–3560.
- X. Tan, Y. Li, J. Liu, and L. Jiang, “Face liveness detection from a
  single image with sparse low rank bilinear discriminative model,” in
  Proceedings of the 11th European Conference on Computer Vision:
  Part VI, ser. ECCV’10. Berlin, Heidelberg: Springer-Verlag, 2010,
  pp. 504–517.
- W. R. Schwartz, A. Rocha, and H. Pedrini, “Face spoofing detection
  through partial least squares and low-level descriptors,” in 2011 International
  Joint Conference on Biometrics (IJCB), Oct 2011, pp. 1–8.
- T. J. d. Carvalho, C. Riess, E. Angelopoulou, H. Pedrini, and
  A. d. R. Rocha, “Exposing digital image forgeries by illumination
  color classification,” IEEE Transactions on Information Forensics and
  Security, vol. 8, no. 7, pp. 1182–1194, July 2013
- R. T. Tan, K. Ikeuchi, and K. Nishino, “Color constancy through inverseintensity
  chromaticity space,” in Digitally Archiving Cultural Objects.
  Springer, 2008, pp. 323–351
- C. Godard, O. Mac Aodha, and G. J. Brostow, “Unsupervised monocular
  depth estimation with left-right consistency,” in CVPR, 2017
- W. Zhu, S. Liang, Y. Wei, and J. Sun, “Saliency optimization from robust
  background detection,” in 2014 IEEE Conference on Computer Vision
  and Pattern Recognition, June 2014, pp. 2814–2821.
- K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
   recognition,” CoRR, vol. abs/1512.03385, 2015.
- J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are
  features in deep neural networks?” CoRR, vol. abs/1411.1792, 2014.
- N. M. Nasrabadi, “Pattern recognition and machine learning,” Journal
  of electronic imaging, vol. 16, no. 4, p. 049901, 2007.
- Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, “A face antispoofing
  database with diverse attacks,” in 2012 5th IAPR International
  Conference on Biometrics (ICB), March 2012, pp. 26–31
- I. Chingovska, A. Anjos, and S. Marcel, “On the effectiveness of local
  binary patterns in face anti-spoofing,” 2012.
- “Information technology – Biometric presentation attack detection – Part
  3: Testing and reporting,” International Organization for Standardization,
  Geneva, CH, Standard, Mar. 2017.
- A. Pinto, H. Pedrini, M. Krumdick, B. Becker, A. Czajka, K. W. Bowyer,
  and A. Rocha, “Counteracting presentation attacks in face, fingerprint,
  and iris recognition,” Deep Learning in Biometrics, p. 245, 2018
- J. Yang, Z. Lei, and S. Z. Li, “Learn Convolutional Neural Network for
  Face Anti-Spoofing,” ArXiv e-prints, Aug. 2014.
- K. Patel, H. Han, and A. K. Jain, “Cross-database face antispoofing
  with robust feature representation,” in Biometric Recognition, Z. You,
  J. Zhou, Y. Wang, Z. Sun, S. Shan, W. Zheng, J. Feng, and Q. Zhao,
  Eds. Cham: Springer International Publishing, 2016, pp. 611–619.
- Deep Learning, An MIT Press book by Ian Goodfellow and Yoshua Bengio and Aaron Courville