# AI_Resources
resources about AI, Machine Learning, Deep Learning, Python...etc

* [Machine Learning](#machine-learning)
  * [Feature Engineering](#feature-engineering)
* [Data Processing](#data-processing)
  * [Text](#text)
      * [Parser](#parser)
      * [Dictionary](#dictionary)
      * [Crawler](#crawler)
      * [Data Augmentation](#data-augmentation)
* [Deep Learning](#deep-learning)
    * [Nets Model](#nets-model)
    * [Optimizer](#optimizer)
    * [Initializer](#initializer)
* [CNN](#cnn)
* [RNN](#rnn)
* [GAN](#gan)
* [NLP](#nlp)
* [RL](#rl)
* [CV](#cv)
   * [Object Detection](#object-detection)
   * [Model Implementation](#model-implementation)
* [Others](#others)
   * [Alpha GO](#alpha-go)
* [Python](#python)
    * [Tensorflow](#tensorflow)


# AI
## Math
[Basic Math Symbols](https://www.rapidtables.com/math/symbols/Basic_Math_Symbols.html)
[Tensor](https://hackernoon.com/learning-ai-if-you-suck-at-math-p4-tensors-illustrated-with-cats-27f0002c9b32)
## Machine Learning
### Courses
[CS109](https://github.com/cs109/2015/tree/master/Lectures)
### Feature Engineering
[特徵工程到底是什麼？](https://www.zhihu.com/question/28641663/answer/110165221?utm_source=com.facebook.katana&utm_medium=social)
## Data Processing
[Why do we noramalize image by subtracting dataset's image mean](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)
### Text
[中文正規式](http://blog.csdn.net/gatieme/article/details/43235791)
#### parser
[jieba 結巴](https://github.com/fxsjy/jieba) : python 中文斷詞套件

[中研院斷詞系統](http://ckipsvr.iis.sinica.edu.tw/): 需申請
#### Dictionary
[LIDC](https://cliwc.weebly.com/liwc.html): 中文詞性、情緒分析

[E-HowNet](http://ehownet.iis.sinica.edu.tw/index.php)：中文詞性、結構化分析、英文對應

[Sinica NLPLab CSentiPackage](http://academiasinicanlplab.github.io/): Java 文章情緒分析
### Word Vector
[Gensim](https://radimrehurek.com/gensim/tutorial.html)

[word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[Glove](https://nlp.stanford.edu/pubs/glove.pdf)
#### crawler
[PTT crawler](https://github.com/afunTW/ptt-web-crawler)
### Data Augmentation
[深度學習中的Data Augmentation方法和代碼實現](https://absentm.github.io/2016/06/14/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84Data-Augmentation%E6%96%B9%E6%B3%95%E5%92%8C%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/)

[The Effectiveness of Data Augmentation in Image Classification using Deep
Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
# Deep Learning
### Nets Model
#### ResNet
* [ResNet：深度殘差網路](https://zh.gluon.ai/chapter_convolutional-neural-networks/resnet-gluon.html)
#### DiracNet
* [對比ResNet： 超深層網絡DiracNet的PyTorch實現](https://www.jiqizhixin.com/articles/2017-11-14-3)
### Optimizer
* [深度學習最全優化方法總結比較（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)
### Initializer
* [Xavier initializer](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
## NLP
* N-Gram
  * [Modeling Natural Language with N-Gram Models](https://sookocheff.com/post/nlp/n-gram-modeling/)
  * [自然語言處理中N-Gram模型介紹](https://zhuanlan.zhihu.com/p/32829048)
* Topic Model
  * [Beginners Guide to Topic Modeling in Python](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)
  * [Topic Modeling with Scikit Learn](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
  * [Latent Dirichlet Allocation](http://ai.stanford.edu/~ang/papers/nips01-lda.pdf)
  * [LDA數學八卦](http://www.victoriawy.com/wp-content/uploads/2017/12/LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf)
  * [手刻板 topic model](https://zhuanlan.zhihu.com/p/23114198)

## CNN
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
* [Best Practices for Document Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)
* [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/pdf/1611.06455.pdf)
* [What does 1x1 convolution mean?](https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network)
* [卷積神經網絡中用1*1 卷積有什麼作用或者好處呢？](https://www.zhihu.com/question/56024942)
* [Transposed Convolution, Fractionally Strided Convolution or Deconvolution](https://buptldy.github.io/2016/10/29/2016-10-29-deconv/)
* [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf): convolution, pooling, stride, transpose convolution
## RNN
* [The Unreasonable Effectiveness of Recurrent Neural Networks by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [LSTM RNN 循環神經網絡(LSTM) by 莫煩](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/2-4-LSTM/)
* [DNN, CNN, RNN 比較](https://www.zhihu.com/question/34681168)
* [Udacity RNN quick introduction](https://www.youtube.com/watch?time_continue=1&v=70MgF-IwAr8)
* [LSTM Networks - The Math of Intelligence](https://www.youtube.com/watch?v=9zhrxE5PQgY) : handcrafted in numpy by Siraj Raval
## GAN
* [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)
* [OpenAI generative model](https://blog.openai.com/generative-models/)
* [DCGAN](https://arxiv.org/abs/1511.06434)
* [Ian Goodfellow's GAN recommendation list:](https://twitter.com/timnitGebru/status/968242968007200769)

  * [Progressive GANs](https://arxiv.org/abs/1710.10196): (probably the highest quality images so far) 
  
  * [Spectral normalization](https://openreview.net/forum?id=B1QRgziT-&noteId=BkxnM1TrM):(got GANs working on lots of classes, which has been hard)
  
  * [Projection discriminator](https://openreview.net/forum?id=ByS1VpgRZ): (from the same lab as #2, both techniques work well together, overall give very good results with 1000 classes) Here’s the video of putting the two methods together: https://www.youtube.com/watch?time_continue=3&v=r6zZPn-6dPY 
  
  * [pix2pixHD](https://arxiv.org/abs/1711.11585) (GANs for 2-megapixel [video](https://www.youtube.com/watch?v=3AIpPlzM_qs&feature=youtu.be )) 
  
  * [Are GANs created equal](https://arxiv.org/abs/1711.10337)?  A big empirical study showing the importance of good rigorous empirical work and how a lot of the GAN variants don’t seem to actually offer improvements in practice
  
  * [WGAN-GP](https://arxiv.org/abs/1704.00028): probably the most popular GAN variant today and seems to be pretty good in my opinion. Caveat: the baseline GAN variants should not perform nearly as badly as this paper claims, especially the text one 
  
  * [StackGAN++](https://arxiv.org/abs/1710.10916): High quality text-to-image synthesis with GANs 
  
  * Making all ML algorithms differentially private by training them on fake private data generated by GANs: https://www.biorxiv.org/content/early/2017/07/05/159756
  
  * You should be a little bit aware of the “GANs with encoders” space, one of my favorites is https://arxiv.org/abs/1701.04722 
  
  * You should be a little bit aware of the “theory of GAN convergence” space, one of my favorites is https://arxiv.org/abs/1706.04156

## RL
* [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [CS 294: Deep Reinforcement Learning, Fall 2017](http://rll.berkeley.edu/deeprlcourse/)
* [DQN從入門到放棄6 DQN的各種改進](https://zhuanlan.zhihu.com/p/21547911)
* [David Silver大神RL](https://www.youtube.com/watch?v=lfHX2hHRMVQ&index=2&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)
* [Applied Deep Learning Machine Learning and Having It Deep and Structured](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/syllabus.html)
* [莫煩RL](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)
* [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
* [李弘毅 A3C](https://www.youtube.com/watch?time_continue=8&v=O79Ic8XBzvw)
* [Denny Britz RL](https://github.com/dennybritz/reinforcement-learning)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [Awesome Reinforcement Learning](https://github.com/aikorea/awesome-rl) by aikorea
* [How to write a reward function](https://www.youtube.com/watch?time_continue=541&v=0R3PnJEisqk) by bonsai

## CV
* [Awesome CV](https://github.com/jbhuang0604/awesome-computer-vision)
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
* [Tool for label data](https://github.com/tzutalin/ImageNet_Utils)
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325): one stage detector
### Object Detection
* [Selective Search for Object Recognition](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)
* [CS231n Lecture 8 - Localization and Detection](https://www.youtube.com/watch?v=_GfPYLNQank&t=574s)
* [RCNN算法詳解](http://blog.csdn.net/shenxiaolu1984/article/details/51066975)
* [RCNN- 將CNN引入目標檢測的開山之作](https://zhuanlan.zhihu.com/p/23006190)
* [原始圖片中的ROI如何映射到到feature map?](https://zhuanlan.zhihu.com/p/24780433)
* [Fast RCNN算法詳解](http://blog.csdn.net/shenxiaolu1984/article/details/51036677)
* [Fast R-CNN Author Slides](http://www.robots.ox.ac.uk/~tvg/publications/talks/fast-rcnn-slides.pdf)
* [Kaming He & RGB: ResNet, R-CNN on CVPR 2017 ](https://www.youtube.com/watch?v=jHv37mKAhV4)
* [目標檢測之RCNN，SPP-NET，Fast-RCNN，Faster-RCNN](http://lanbing510.info/2017/08/24/RCNN-FastRCNN-FasterRCNN.html)
* [Keras on Faster R-CNN](https://zhuanlan.zhihu.com/p/28585873)
* [How does the region proposal network (RPN) in Faster R-CNN work?](https://www.quora.com/How-does-the-region-proposal-network-RPN-in-Faster-R-CNN-work)
* [AP: Average Precision](https://sanchom.wordpress.com/tag/average-precision/)
* [Light head R-CNN](https://arxiv.org/abs/1711.07264)
### Segmentation
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211) paper
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) paper
### Model Implementation
* [R-CNN](https://github.com/rbgirshick/rcnn)
* [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn)
* [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn)
* [Mask R-CNN](http://forums.fast.ai/t/implementing-mask-r-cnn/2234)
* [Yolo Paper](https://arxiv.org/abs/1506.02640)
* [Yolo 9000 Paper](https://arxiv.org/abs/1612.08242)
* [Project Yolo](https://pjreddie.com/darknet/yolo/)
* [YOLO9000: Better, Faster, Stronger論文筆記](https://www.jianshu.com/p/2d88bdd89ba0)
* [YOLO2 - YAD2K](https://sanchom.wordpress.com/tag/average-precision/)
### Dataset
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/): 20 classes
  * [tutorial](http://blog.csdn.net/weixin_35653315/article/details/71028523)
* [Miscrosoft COCO](http://cocodataset.org/#home): 80 classes
  * [tutorial](http://blog.csdn.net/u012905422/article/details/52372755)
### AOI
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf)
## Others
* [Machine Learning for Systems and Systems for Machine Learning](http://learningsys.org/nips17/assets/slides/dean-nips17.pdf): by Jeff Dean
### Alpha GO
* [Alpha Zero Cheetsheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
* [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)

## Python
### Tensorflow
[莫煩 Tensowflow 教學](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/)

[Slim 介紹](http://blog.csdn.net/mao_xiao_feng/article/details/73409975)

[Slim Nets List](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets)
### Numpy
[100 numpy exercises](https://github.com/rougier/numpy-100/blob/master/100%20Numpy%20exercises.md)
