# nodule_segmentation
这个里面有两个model，一个是原来的unet，也是我在推送里提到的unet，这个网络再训练的时候没有收敛，所以我们找了一个预训练模型，就是pre_trained_unet，
有已经训练好的模型，我就在这个基础上finetune
数据接口都是针对公司服务器路径写的，有需要可以自己更改接口
我在这个程序里第一次实现了validation，在dataset里用了tf.data这个模块，是真的好用，强烈建议大家学一学……
