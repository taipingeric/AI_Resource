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
* [GAN](#gan)
* [Python](#python)
    * [Tensorflow](#tensorflow)


# AI
## Machine Learning
### Feature Engineering
[特徵工程到底是什麼？](https://www.zhihu.com/question/28641663/answer/110165221?utm_source=com.facebook.katana&utm_medium=social)
## Data Processing
[Why do we noramalize image by subtracting dataset's image mean](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)
### Text
#### parser
[jieba 結巴](https://github.com/fxsjy/jieba) : python 中文斷詞套件

[中研院斷詞系統](http://ckipsvr.iis.sinica.edu.tw/): 需申請
#### Dictionary
[LIDC](https://cliwc.weebly.com/liwc.html): 中文詞性、情緒分析

[E-HowNet](http://ehownet.iis.sinica.edu.tw/index.php)：中文詞性、結構化分析、英文對應

[Sinica NLPLab CSentiPackage](http://academiasinicanlplab.github.io/): Java 文章情緒分析
#### crawler
[PTT crawler](https://github.com/afunTW/ptt-web-crawler)
### Data Augmentation
[深度學習中的Data Augmentation方法和代碼實現](https://absentm.github.io/2016/06/14/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84Data-Augmentation%E6%96%B9%E6%B3%95%E5%92%8C%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/)

[The Effectiveness of Data Augmentation in Image Classification using Deep
Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
## Deep Learning
### Nets Model
#### ResNet
* [ResNet：深度殘差網路](https://zh.gluon.ai/chapter_convolutional-neural-networks/resnet-gluon.html)
#### DiracNet
* [對比ResNet： 超深層網絡DiracNet的PyTorch實現](https://www.jiqizhixin.com/articles/2017-11-14-3)
### Optimizer
* [深度學習最全優化方法總結比較（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)
## GAN
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

## Python
### Tensorflow
[Slim 介紹](http://blog.csdn.net/mao_xiao_feng/article/details/73409975)

[Slim Nets List](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets)
