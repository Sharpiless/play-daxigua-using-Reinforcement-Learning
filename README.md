
# 用强化学习玩合成大西瓜

代码地址：[https://github.com/Sharpiless/play-daxigua-using-Reinforcement-Learning](https://github.com/Sharpiless/play-daxigua-using-Reinforcement-Learning)

用强化学习DQN算法，训练AI模型来玩合成大西瓜游戏，提供Keras版本、PARL（paddle）版本和pytorch版本。

> B站：[https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)

> CSDN：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)

> AI Studio：[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> Github：[https://github.com/Sharpiless](https://github.com/Sharpiless)

## 1. 打开游戏：

这里使用pygame重写了大西瓜游戏，并封装为适合RL环境的代码。

运行：

```bash
python Main.py
```

即可开始游戏：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212172120818.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

## 2. 训练RL模型：

RL算法采用DQN算法，其中Keras版本使用了简单的卷积神经网络来计算Q值，PRAL版本使用ResNet。

运行：

```bash
python train_keras.py
```

或者

```bash
python train_paddle.py
```

或者

```bash
python train_torch.py
```

开始训练：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210212172442170.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

# 关注我的公众号：

感兴趣的同学关注我的公众号——可达鸭的深度学习教程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210127153004430.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

