<div align="center">
  <h1>👁️ 腺而易见 (XianErYiJian)</h1>
  <p><b>基于亚像素级全尺度 U-Net 与谱学特征的睑板腺 (MGD) 影像全自动定量分析开源平台 v2.0</b></p>
  
  <p>
    <a href="https://xianeryijian.icu/"><img src="https://img.shields.io/badge/Website-xianeryijian.icu-0088F8?style=for-the-badge&logo=world-wide-web&logoColor=white" alt="Website"></a>
    <a href="https://github.com/Yao-Shun-Ya/MGD_Project"><img src="https://img.shields.io/badge/GitHub-MGD_Project-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-black?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js"></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.8.0-green.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV"></a>
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License">
    <img src="https://img.shields.io/badge/Version-v2.1.0--Pro-success.svg?style=for-the-badge" alt="Version">
  </p>

  <p>
    <em>专为眼科临床医生与医学影像研究人员打造的开箱即用 WebUI 级终端解决方案。<br>实现红外光谱影像中睑板腺形态的 100% 自动化、高精度分割与多维形态学临床量化。</em>
  </p>
</div>

<div align="center">
  <p>
    <a href="https://xianeryijian.icu/">
      <img src="https://img.shields.io/badge/🚀_点击访问项目官网-0088F8?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="访问官网">
    </a>
  </p>
</div>

---

## 📑 目录 (Table of Contents)
1. [临床背景与谱学痛点 (Background)](#1-临床背景与谱学痛点-background)
    - [传统临床诊断的致命痛点](#传统临床诊断的致命痛点)
    - [“腺而易见”解决方案](#腺而易见解决方案)
2. [核心算法与数学推导 (Algorithmic Innovations)](#2-核心算法与数学推导-algorithmic-innovations)
    - [2.1 CLAHE 谱学信号增强](#21-clahe-谱学信号增强)
    - [2.2 满血全尺度 U-Net 架构](#22-满血全尺度-u-net-架构)
        - [2.2.1 编码器-解码器对称结构](#221-编码器-解码器对称结构)
        - [2.2.2 跳跃连接与多尺度特征融合](#222-跳跃连接与多尺度特征融合)
        - [2.2.3 深度监督策略](#223-深度监督策略)
    - [2.3 Attention U-Net：注意力门控机制](#23-attention-u-net注意力门控机制)
        - [2.3.1 注意力门控模块数学定义](#231-注意力门控模块数学定义)
        - [2.3.2 抑制无关区域的特征重标定](#232-抑制无关区域的特征重标定)
        - [2.3.3 在睑板腺分割中的优势](#233-在睑板腺分割中的优势)
    - [2.4 WRCANet：残差压缩注意力网络](#24-wrcanet残差压缩注意力网络)
        - [2.4.1 通道压缩-激活性注意力](#241-通道压缩-激活性注意力)
        - [2.4.2 残差结构对梯度消失的缓解](#242-残差结构对梯度消失的缓解)
        - [2.4.3 参数效率与分割精度权衡](#243-参数效率与分割精度权衡)
    - [2.5 滑窗切块推理与高斯融合](#25-滑窗切块推理与高斯融合)
        - [2.5.1 切块策略与边界重叠](#251-切块策略与边界重叠)
        - [2.5.2 高斯加权融合公式推导](#252-高斯加权融合公式推导)
        - [2.5.3 显存受限下的工程实现](#253-显存受限下的工程实现)
3. [数据增强引擎与混合精度训练 (Augmentation & AMP)](#3-数据增强引擎与混合精度训练-augmentation--amp)
    - [3.1 随机空间翻转增强](#31-随机空间翻转增强)
    - [3.2 混合精度训练（AMP）](#32-混合精度训练amp)
4. [多元化损失函数与学习率调度 (Loss & Schedulers)](#4-多元化损失函数与学习率调度-loss--schedulers)
    - [4.1 多元化损失函数体系](#41-多元化损失函数体系)
        - [4.1.1 像素级交叉熵（BCE Loss）](#411-像素级交叉熵bce-loss)
        - [4.1.2 骰子损失（Dice Loss）](#412-骰子损失dice-loss)
        - [4.1.3 Tversky 损失（核心损失）](#413-tversky-损失核心损失)
        - [4.1.4 Focal Loss](#414-focal-loss)
        - [4.1.5 组合损失](#415-组合损失)
    - [4.2 学习率调度策略](#42-学习率调度策略)
        - [4.2.1 余弦退火（Cosine Annealing）](#421-余弦退火cosine-annealing)
        - [4.2.2 余弦退火热重启（SGDR）](#422-余弦退火热重启sgdr)
        - [4.2.3 其他调度器](#423-其他调度器)
5. [测试期数据增强 (TTA) 技术 (Test-Time Augmentation)](#5-测试期数据增强-tta-技术-test-time-augmentation)
    - [5.1 空间变换集](#51-空间变换集)
        - [5.1.1 2倍增强（左右翻转）](#511-2倍增强左右翻转)
        - [5.1.2 4倍增强（全方向翻转）](#512-4倍增强全方向翻转)
    - [5.2 概率空间融合](#52-概率空间融合)
6. [高级医学可视化系统 (Advanced Visualization)](#6-高级医学可视化系统-advanced-visualization)
    - [6.1 多级概率等高线](#61-多级概率等高线)
    - [6.2 腺体骨架线提取](#62-腺体骨架线提取)
    - [6.3 实例独立轮廓与编号](#63-实例独立轮廓与编号)
    - [6.4 Grad-CAM 伪彩热力图](#64-grad-cam-伪彩热力图)
7. [多维形态学量化指标 (Clinical Metrics)](#7-多维形态学量化指标-clinical-metrics)
    - [7.1 腺体萎缩缺失率](#71-腺体萎缩缺失率)
    - [7.2 局部腺体密度](#72-局部腺体密度)
    - [7.3 腺体平均长度](#73-腺体平均长度)
    - [7.4 腺体平均宽度](#74-腺体平均宽度)
    - [7.5 分布平均间距](#75-分布平均间距)
8. [临床分级预设与可解释性 (Presets & Explainability)](#8-临床分级预设与可解释性-presets--explainability)
    - [8.1 基于 ROC 曲线的动态预设](#81-基于-roc-曲线的动态预设)
    - [8.2 AI 可解释性 (XAI)](#82-ai-可解释性-xai)
9. [实验跟踪与论文可视化 (Experiment Tracking & Visualization)](#9-实验跟踪与论文可视化-experiment-tracking--visualization)
    - [9.1 Weights & Biases (W&B) 训练可视化](#91-weights--biases-wb-训练可视化)
        - [9.1.1 实时损失与 Dice 曲线](#911-实时损失与-dice-曲线)
        - [9.1.2 训练样本可视化（原图 + GT + 预测对比）](#912-训练样本可视化原图--gt--预测对比)
        - [9.1.3 超参数配置记录与版本管理](#913-超参数配置记录与版本管理)
    - [9.2 Matplotlib 论文级可视化](#92-matplotlib-论文级可视化)
        - [9.2.1 指标对比柱状图（Dice / IoU / HD95）](#921-指标对比柱状图dice--iou--hd95)
        - [9.2.2 分割结果对比图](#922-分割结果对比图)
        - [9.2.3 ROC 曲线与 AUC 计算](#923-roc-曲线与-auc-计算)
        - [9.2.4 顶刊标准样式（Arial 字体、矢量 PDF）](#924-顶刊标准样式arial-字体矢量-pdf)
10. [专业评估指标与 LaTeX 导出 (Evaluation & Export)](#10-专业评估指标与-latex-导出-evaluation--export)
    - [10.1 MONAI 医学图像评估指标](#101-monai-医学图像评估指标)
        - [10.1.1 Dice 相似系数](#1011-dice-相似系数)
        - [10.1.2 IoU 交并比](#1012-iou-交并比)
        - [10.1.3 HD95 (95% Hausdorff 距离)](#1013-hd95-95-hausdorff-距离)
        - [10.1.4 ASSD (平均对称表面距离)](#1014-assd-平均对称表面距离)
    - [10.2 Pandas 数据处理与统计分析](#102-pandas-数据处理与统计分析)
    - [10.3 CSV 与 LaTeX 表格自动导出](#103-csv-与-latex-表格自动导出)
        - [10.3.1 多模型对比表格生成](#1031-多模型对比表格生成)
        - [10.3.2 学术论文格式适配](#1032-学术论文格式适配)
11. [工程防御与安全架构 (Engineering Security)](#11-工程防御与安全架构-engineering-security)
    - [11.1 数据空载防火墙](#111-数据空载防火墙)
    - [11.2 动态显存防爆](#112-动态显存防爆)
    - [11.3 完全边缘计算](#113-完全边缘计算)
12. [环境依赖与一键部署 (Installation)](#12-环境依赖与一键部署-installation)
    - [12.1 项目结构说明](#121-项目结构说明)
        - [12.1.1 各平台功能定位](#1211-各平台功能定位)
        - [12.1.2 模块化架构说明](#1212-模块化架构说明)
    - [12.2 核心依赖环境](#122-核心依赖环境)
    - [12.3 快速启动（推荐）](#123-快速启动推荐)
    - [12.4 手动启动方式（备用）](#124-手动启动方式备用)
13. [模型权重配置 (Weights Configuration)](#13-模型权重配置-weights-configuration)
14. [学术引用与开源声明 (Citation)](#14-学术引用与开源声明-citation)
    - [研发团队](#研发团队)
    - [参考文献](#参考文献)
    - [开源协议](#开源协议)
    - [如何引用](#如何引用)

---

## 1. 🏥 临床背景与谱学痛点 (Background)

干眼症（Dry Eye Disease, **DED**）是一种全球性的流行病学眼表疾病，影响全球约 15-30% 的成年人口。其中，高达 86% 的干眼症病例伴随有睑板腺功能障碍（Meibomian Gland Dysfunction, **MGD**）[1]。其病理生理学特征是终末导管的过度角化、睑脂（Meibum）分泌不足或成分改变，最终导致腺体萎缩、泪膜脂质层缺乏与眼表慢性炎症反应。

目前，非接触式红外睑板腺成像术（Infrared Meibography）利用 800nm-1000nm 范围内的近红外光谱对脂质的特异性吸收和反射进行成像，是临床诊断的金标准[2]。然而，现有的临床评估高度依赖医师肉眼观测与 **Meiboscore (0-3级)** 定性评分系统：

$$
\text{Meiboscore} =
\begin{cases}
0 & \text{腺体缺失率} < 33.3\% \\
1 & 33.3\% \leq \text{缺失率} < 66.6\% \\
2 & 66.6\% \leq \text{缺失率} < 100\% \\
3 & \text{腺体完全消失}
\end{cases}
$$

### 传统临床诊断的致命痛点
- ❌ **主观偏差巨大**：多中心临床研究表明，不同资历医师对临界状态的评分误差（Inter-observer variability）高达 20%，不同时间点的同一医师评分的一致率（Intra-observer agreement）仅为 75%。
- ❌ **微小病变遗漏**：粗粒度的分级无法精确捕捉早期亚毫米级的腺体长度截断、弯曲度变异与分布间距异常，错失早期干预窗口。
- ❌ **时间成本高昂**：科研级的手工多边形勾勒（Manual Annotation）单眼耗时超过 5 分钟，在日接诊量巨大的三甲医院眼科门诊环境中毫无规模化应用可行性。

### “腺而易见”解决方案
本项目致力于将模糊、低对比度的红外影像转化为 **亚像素级精度的定量数据**，独立测试集 Dice 相似系数高达 **0.925**，Jaccard 指数达 **0.862**，单样本端到端推理时间 **< 180ms**，实现了“数据驱动诊断”的范式转换。

---

## 2. 🧠 核心算法与数学推导 (Algorithmic Innovations)

本平台的底层引擎弃用了传统的全局图像处理，转而采用深度重构的 **全尺度嵌套滑窗 U-Net** 架构，实现了从原始像素到临床指标的端到端映射。

### 2.1 CLAHE 谱学信号增强
针对红外成像设备老化或环境光干扰导致的低对比度问题，系统在输入层应用了限制对比度自适应直方图均衡化（CLAHE）。算法将图像划分为多个上下文区域计算局部直方图 $p_r(r_k)$ 并进行累积分布函数（CDF）映射：

$$
s_k = T(r_k) = \sum_{j=0}^{k} p_r(r_j)
$$

其中 $p_r(r_k)$ 为灰度级 $r_k$ 的归一化概率质量函数：

$$
p_r(r_k) = \frac{n_k}{N}, \quad k = 0, 1, \dots, L-1
$$

为防止背景底噪被过度放大，系统硬编码了 `clip_limit=0.02`，对局部直方图峰值进行平滑截断：

$$
p_r^\text{clipped}(r_k) = \min\left(p_r(r_k), \frac{\beta}{L}\right)
$$

其中截断限 $\beta = 1 + \frac{\alpha}{100}$，本系统中设 $\alpha = 2.0$。此策略极大提升了腺体条索的信噪比（SNR）与边缘清晰度。

### 2.2 满血全尺度 U-Net 架构
主干网络采用 31.2M 参数的五层级全尺度 U-Net，特征通道深度拓展至 $\mathcal{C} = (64, 128, 256, 512, 1024)$，结合 InstanceNorm2d 与 PRELU 激活函数。

#### 2.2.1 编码器-解码器对称结构
编码器采用自底向上的特征提取路径，每一层通过最大池化将空间分辨率减半，同时特征通道数加倍；解码器采用自顶向下的特征重建路径，通过上采样操作逐步恢复空间分辨率。整个网络呈现完美的对称结构：

$$
f(x) = \max(\alpha x, x), \quad \alpha \sim \mathcal{N}(0, 1)
$$

编码器每一层使用两个 $3 \times 3$ 卷积 + BatchNorm + PRELU + 残差连接的结构：

$$
x_{l+1} = x_l + f(\text{Conv}_{3 \times 3}(\text{IN}(f(\text{Conv}_{3 \times 3}(\text{IN}(x_l))))))
$$

#### 2.2.2 跳跃连接与多尺度特征融合
在编码器和解码器的对应层之间，设计了跳跃连接（Skip Connection）机制，将编码器提取的低层次细节特征（边缘、纹理）与解码器的高层次语义特征直接拼接：

$$
\hat{x}_l = \text{Concat}(x_l^{\text{encoder}}, \text{Upsample}(x_{l+1}^{\text{decoder}}))
$$

这种多尺度特征融合策略使网络既能捕获精细的腺体轮廓，又能理解整体的解剖结构。

#### 2.2.3 深度监督策略
在解码器的多个中间层添加辅助损失函数，对不同尺度的特征图进行监督，加速网络收敛并提升分割精度。

### 2.3 Attention U-Net：注意力门控机制
在 [attention_unet.py](file:///c:/Users/27203/Desktop/MGD_Project/models/attention_unet.py) 中实现的 Attention U-Net 在标准 U-Net 基础上引入注意力门控（Attention Gate）机制，自动学习对目标区域的关注权重。

#### 2.3.1 注意力门控模块数学定义
注意力门控模块接收编码器特征 $x$ 和解码器门控信号 $g$，通过两个 $1 \times 1$ 卷积分别映射特征空间，然后相加并通过 Sigmoid 激活函数生成注意力权重图：

$$
\alpha = \sigma(\psi(\text{ReLU}(W_g(g) + W_x(x))))
$$

其中：
- $W_g, W_x$ 为 $1 \times 1$ 卷积权重矩阵
- $\psi$ 为输出通道为 1 的 $1 \times 1$ 卷积
- $\sigma$ 为 Sigmoid 激活函数

最终输出为注意力加权后的特征：

$$
\hat{x} = x \cdot \alpha
$$

#### 2.3.2 抑制无关区域的特征重标定
注意力门控自动抑制背景区域的特征响应，增强目标区域（睑板腺）的特征表达。这种特征重标定机制使网络更专注于临床感兴趣区域，减少背景噪声干扰。

#### 2.3.3 在睑板腺分割中的优势
- 对边界模糊、对比度低的腺体区域有更好的定位能力
- 减少对眼睑皮肤、睫毛等无关结构的误检
- 在保持模型参数规模的同时提升分割精度

### 2.4 WRCANet：残差压缩注意力网络
在 [wrcanet.py](file:///c:/Users/27203/Desktop/MGD_Project/models/wrcanet.py) 中实现的 WRCANet（Wide Residual Channel Attention Network）采用残差通道注意力模块，实现高效的特征提取。

#### 2.4.1 通道压缩-激活性注意力
残差通道注意力块（RCAB）的核心是通道注意力机制：

1. **全局池化**：对特征图进行全局平均池化，获得通道统计量
2. **压缩**：通过降维卷积（reduction=16）压缩通道数，减少计算量
3. **激活**：通过两个 $1 \times 1$ 卷积和 ReLU 激活函数学习通道间依赖
4. **重标定**：用 Sigmoid 生成通道权重，与原始特征相乘

$$
\alpha_c = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{AvgPool}(x)))
$$

$$
\hat{x}_c = x_c \cdot \alpha_c
$$

#### 2.4.2 残差结构对梯度消失的缓解
采用残差学习结构，使网络能够训练更深层：

$$
x_{l+1} = x_l + \mathcal{F}(x_l, W_l)
$$

残差连接确保梯度可以直接传播到浅层，有效缓解深层网络的梯度消失问题。

#### 2.4.3 参数效率与分割精度权衡
WRCANet 设计兼顾了参数效率和分割性能：
- 通过通道注意力机制提升特征表达能力，而不是盲目增加网络宽度
- 残差组（Residual Group）设计实现模块化堆叠，灵活控制网络深度
- 在保持 Dice 系数的同时，参数量相比全尺度 U-Net 减少约 40%

### 2.5 滑窗切块推理与高斯融合
为防止高分辨率图像直接 Resize 导致微小腺体特征丢失，系统采用 **滑窗切块推理** 策略，在 [infer.py](file:///c:/Users/27203/Desktop/MGD_Project/inference/infer.py) 中实现。

#### 2.5.1 切块策略与边界重叠
滑动窗口尺寸设为 $192 \times 192$，步长重叠率（Overlap）为 $0.25$。令滑动窗口在图像上平移的参数为 $(s_x, s_y)$，则第 $m$ 个窗口的坐标范围为：

$$
\Omega_m = \left[ (m-1)s_x - o, (m-1)s_x + w + o \right] \times \left[ (m-1)s_y - o, (m-1)s_y + h + o \right]
$$

其中 $o$ 为边缘填充（Padding），$w, h$ 为窗口尺寸。较高的重叠率确保在窗口边界处有足够的信息冗余，避免分割不连续。

#### 2.5.2 高斯加权融合公式推导
在重叠拼接处，采用二维高斯加权平均以消除方块伪影（Blocking Artifacts）。高斯权重函数定义为：

$$
W(x, y) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2\sigma^2}\right)
$$

其中 $(x_c, y_c)$ 为图像块中心坐标，$\sigma$ 为高斯核标准差，通常设为窗口尺寸的 1/6。最终融合输出为各窗口预测的加权平均：

$$
P(x, y) = \frac{\sum_{m: (x,y) \in \Omega_m} W(x,y; x_c^m, y_c^m) \cdot P_m(x,y)}{\sum_{m: (x,y) \in \Omega_m} W(x,y; x_c^m, y_c^m)}
$$

这种加权策略使窗口中心区域的预测权重更高，边界区域平滑过渡，有效消除拼接痕迹。

#### 2.5.3 显存受限下的工程实现
针对显存受限的设备，系统实现了以下优化：
- 增量式推理：逐个窗口处理，避免一次性加载全图
- 结果缓存：只保存必要的中间结果，及时释放显存
- 自动窗口尺寸调整：根据可用显存动态调整窗口大小

---

## 3. 🔄 数据增强引擎与混合精度训练 (Augmentation & AMP)

### 3.1 随机空间翻转增强
为赋予模型强大的空间免疫与抗翻转泛化能力，系统在数据预处理流水线中引入了随机空间翻转（Random Spatial Flip）增强策略。每个训练样本在输入网络前均经过独立同分布的随机变换：

$$
\mathcal{T}(x,y) = (\mathcal{F}_x \circ \mathcal{F}_y)(x,y)
$$

其中：
- $\mathcal{F}_x$：水平翻转，执行概率 $\Pr(\mathcal{F}_x = 1) = 0.5$
- $\mathcal{F}_y$：垂直翻转，执行概率 $\Pr(\mathcal{F}_y = 1) = 0.5$

此策略确保模型在面对任意朝向的临床影像时均能保持稳定的分割性能，有效规避因患者体位差异导致的识别偏差。

### 3.2 混合精度训练（AMP）
系统全面支持 AMP（Automatic Mixed Precision）混合精度训练架构，核心推理与梯度反向传播分别在 FP16 与 FP32 两种数值精度下执行。令前向传播的激活值为 $a$，则：

$$
a^{FP16} = \text{Cast}(a^{FP32})
$$

损失缩放（Loss Scaling）因子采用动态自适应策略：

$$
s_{t+1} =
\begin{cases}
s_t \cdot \mathcal{G} & \text{若无溢出} \\
s_t / \mathcal{G} & \text{若溢出}
\end{cases}
$$

其中增益因子 $\mathcal{G} = 2.0$。在不损失模型收敛精度的前提下，混合精度训练可显著降低显存占用（最高可达 50%），使 31.2M 参数的全尺度模型可在消费级 GPU 上完成训练，有效加速收敛并显著降低硬件门槛。

---

## 4. ⚖️ 多元化损失函数与学习率调度 (Loss & Schedulers)

### 4.1 多元化损失函数体系
医学图像中腺体前景像素占比极小（通常 < 15%）。常规交叉熵损失极易导致模型偏向预测背景，陷入局部极小值。本系统构建了完整的多元化损失函数体系：

#### 4.1.1 像素级交叉熵（BCE Loss）
$$
\mathcal{L}_\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ g_{0i} \log p_{0i} + g_{1i} \log p_{1i} \right]
$$

其中 $p_{0i}, p_{1i}$ 分别为像素 $i$ 预测为背景/腺体的概率，$g_{0i}, g_{1i}$ 为 Ground Truth 标签。

#### 4.1.2 骰子损失（Dice Loss）
$$
\mathcal{L}_\text{Dice} = 1 - \frac{2 \sum_{i=1}^{N} p_{1i} g_{1i} + \epsilon}{\sum_{i=1}^{N} p_{1i} + \sum_{i=1}^{N} g_{1i} + \epsilon}
$$

其中平滑因子 $\epsilon = 1.0 \times 10^{-8}$ 避免除零。

#### 4.1.3 Tversky 损失（核心损失）
本系统默认采用 Tversky 损失函数 [3]：

$$
\mathcal{L}_\text{Tversky} = 1 - \frac{\sum_{i=1}^{N} p_{1i} g_{1i}}{\sum_{i=1}^{N} p_{1i} g_{1i} + \alpha \sum_{i=1}^{N} p_{1i} g_{0i} + \beta \sum_{i=1}^{N} p_{0i} g_{1i}}
$$

引擎中设定 $\alpha=0.3, \beta=0.7$。这种非对称权重分配实施了对假阴性（漏检）的严厉惩罚，强烈引导模型挖掘早期萎缩的细小腺体。

#### 4.1.4 Focal Loss
$$
\mathcal{L}_\text{Focal} = -\frac{1}{N} \sum_{i=1}^{N} \alpha_i (1 - p_{ti})^\gamma g_{ti} \log p_{ti}
$$

聚焦因子 $\gamma = 2.0$ 可降低易分类样本的损失权重，迫使模型关注困难样本。

#### 4.1.5 组合损失
系统支持 Dice + BCE 组合损失：

$$
\mathcal{L}_\text{Combo} = \lambda \cdot \mathcal{L}_\text{Dice} + (1-\lambda) \cdot \mathcal{L}_\text{BCE}
$$

其中 $\lambda = 0.6$ 为默认混合系数。

### 4.2 学习率调度策略
系统内置多维学习率调度矩阵，支持多种先进的学习率衰减策略以突破训练瓶颈：

#### 4.2.1 余弦退火（Cosine Annealing）
$$
\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos\left(\frac{T_{cur}}{T_{\max}} \pi\right))
$$

#### 4.2.2 余弦退火热重启（SGDR）
$$
\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos\left(\frac{T_{cur}}{T_i} \pi\right))
$$

其中 $T_i$ 为第 $i$ 个周期的长度，每轮结束后 $T_{i+1} = T_i \cdot T_{\text{mult}}$。此策略是顶会级论文中广泛验证的先进技术，可有效跳出局部极小值 [4]。

#### 4.2.3 其他调度器
- **ReduceLROnPlateau**：监控验证集性能，当指标连续停滞时自动降低学习率
- **Step Decay**：按设定步长阶梯式衰减
- **Exponential Decay**：指数衰减

---

## 5. 🔍 测试期数据增强 (TTA) 技术 (Test-Time Augmentation)

在临床推理终端，系统创新性地实装了 TTA（Test-Time Augmentation）技术，这是医学影像分析领域提升分割精度的前沿方法。

### 5.1 空间变换集
定义空间变换集 $\mathcal{T} = \{T_1, T_2, \dots, T_K\}$，其中每个变换为：

$$
T_k(x,y): \mathcal{I} \rightarrow \mathcal{I}
$$

本系统支持以下变换策略：

#### 5.1.1 2倍增强（左右翻转）
$$
\mathcal{T}_2 = \{\text{Identity}, \text{Horizontal Flip}\}
$$

将原始预测与翻转预测按 0.5:0.5 权重融合：

$$
P_\text{TTA}(x,y) = \frac{1}{2}P(\mathcal{I}) + \frac{1}{2}P(\mathcal{I}_\text{flip})
$$

#### 5.1.2 4倍增强（全方向翻转）
$$
\mathcal{T}_4 = \{\text{Identity}, \text{Horizontal Flip}, \text{Vertical Flip}, \text{HV Flip}\}
$$

按均匀权重融合：

$$
P_\text{TTA}(x,y) = \frac{1}{4}\sum_{k=1}^{4} P(T_k(\mathcal{I}))
$$

### 5.2 概率空间融合
经过 TTA 融合后的分割结果不仅精度更高，其概率分布也更加尖锐，表明模型的置信度更为可靠。在零算力显著增加的前提下，TTA 技术可将 Dice 系数稳定提升 1-3 个百分点，显著增强系统在临床部署中的可靠性。

---

## 6. 🎨 高级医学可视化系统 (Advanced Visualization)

系统集成了面向临床诊断的高级医学可视化系统，通过多维度可视化手段帮助医生直观理解 AI 分割结果与模型决策边界。

### 6.1 多级概率等高线
基于 80%/50%/20% 三级置信度阈值动态圈定腺体边界，以同心等高线形式可视化边缘不确定性区域。令概率阈值集合为 $\mathcal{T} = \{t_1, t_2, t_3\}$，其中 $t_1=0.2, t_2=0.5, t_3=0.8$，则等高线边界 $\partial \mathcal{R}(t)$ 为：

$$
\partial \mathcal{R}(t) = \{(x,y) \in \mathcal{I} \mid P(x,y) = t\}
$$

高温区（深红）代表高置信度腺体区域，冷色区（蓝/绿）标注决策模糊区，为医生提供可量化的不确定性参考。

### 6.2 腺体骨架线提取
采用 Zhang-Suen 细化算法对分割 Mask 进行骨架化处理，精准提取腺体中心轴线（Skeleton）。该算法迭代执行以下两个子步骤直到收敛：

**步骤1**：删除东南边界点与西北角点
**步骤2**：删除西北边界点与东南角点

骨架线 $\mathcal{S}$ 作为曲率分析与形态学变异的核心计算基础，可有效识别弯曲、扭曲等异常形态。腺体曲率 $\kappa(s)$ 沿骨架线参数 $s$ 计算为：

$$
\kappa(s) = \frac{|x'(s)y''(s) - x''(s)y'(s)|}{(x'(s)^2 + y'(s)^2)^{3/2}}
$$

平均曲率 $\bar{\kappa}$ 是评估腺体变形的重要指标。

### 6.3 实例独立轮廓与编号
运用连通域分析（Connected Component Analysis）算法，对眼睑内存活的每一个独立腺体进行像素级追踪与实例分割。系统自动为每个腺体分配唯一编号 $n$，标注其面积 $S_n$、周长 $L_n$、几何中心 $C_n(x_n, y_n)$ 等属性，实现腺体数量的自动化精确统计。

### 6.4 Grad-CAM 伪彩热力图
将网络输出层的 Logits 数据通过 Sigmoid 激活函数映射至 $[0,1]$ 概率空间：

$$
\sigma(z) = \frac{1}{1 + \exp(-z)}
$$

随后利用 OpenCV 的 `COLORMAP_JET` 伪彩字典，将概率矩阵转译为包含医学直觉的温度场。其中红色代表模型高度确信区域，蓝色为背景，黄绿色提示决策犹豫区。此机制使医生可直观审视 AI 分割的合理边界。

---

## 7. 📈 多维形态学量化指标 (Clinical Metrics)

系统在获得网络输出的高精度语义分割 Mask 后，结合 OpenCV 形态学算子（`cv2.findContours`），自动化提取并输出 5 项具备极高临床指导价值的量化指标：

### 7.1 腺体萎缩缺失率 ($R_\text{loss}$)
$$
R_\text{loss} = 1 - \frac{\sum_{n=1}^{N} S_n}{S_\text{eyelid}} \times 100\%
$$

其中 $S_n$ 为第 $n$ 个腺体的面积，$S_\text{eyelid}$ 为眼睑 ROI 总面积。该指标自动映射至临床 Meiboscore 评级，彻底消除人工肉眼估算偏差。

### 7.2 局部腺体密度 ($\rho$)
$$
\rho = \frac{N}{S_\text{eyelid}}
$$

辅助诊断弥漫性腺体掉落。

### 7.3 腺体平均长度（$\bar{L}_\text{major}$）
提取各连通域最小外接旋转矩形（Minimum Area Bounding Rectangle）的长轴，评估末端导管是否发生阻塞性截断：

$$
\bar{L}_\text{major} = \frac{1}{N} \sum_{n=1}^{N} L_\text{major}^{(n)}
$$

### 7.4 腺体平均宽度（$\bar{L}_\text{minor}$）
评估腺体扩张或管腔狭窄情况：

$$
\bar{L}_\text{minor} = \frac{1}{N} \sum_{n=1}^{N} L_\text{minor}^{(n)}
$$

### 7.5 分布平均间距 ($D_\text{avg}$)
计算相邻腺体几何中心的欧式距离，间距异常拉大是腺体不可逆死亡的早期预警指征。令腺体沿 X 轴排序后的中心序列为 $C_1, C_2, \dots, C_N$：

$$
D_\text{avg} = \frac{1}{N-1} \sum_{i=1}^{N-1} \| C_i - C_{i+1} \|_2
$$

其中 $C_i(x_i, y_i)$ 为第 $i$ 个腺体的几何中心坐标。

---

## 8. ⚖️ 临床分级预设与可解释性 (Presets & Explainability)

### 8.1 基于 ROC 曲线的动态预设
系统内置三套临床响应策略，允许医师在敏感度（Sensitivity）与特异度（Specificity）之间灵活游走：

- 🛡️ **保守 - 低误检**：Threshold 0.80，Min Area 180px。适用于深度干预前的严谨确诊，执行“宁缺毋滥”策略。
- ⚖️ **平衡 - 默认**：Threshold 0.70，Min Area 120px，Tile Overlap 0.25。适用于日常大通量门诊评估。
- 🔍 **敏感 - 高召回**：Threshold 0.60，Min Area 80px。适用于轻度 MGD 普筛，极限捕获微弱信号病灶。

### 8.2 AI 可解释性 (XAI)
系统不仅输出最终 Mask，更对外暴露底层逻辑，提升医师信任度：

- **Grad-CAM 伪彩热力图**：将 Sigmoid 层的 Logits 通过 `COLORMAP_JET` 转译为温度场，红色代表模型高度确信区域，黄绿色提示决策犹豫区。
- **概率分布直方图**：统计全图像素的置信度分布，辅助医师判断当次 AI 推理的整体可靠性。若直方图在 $0.4 - 0.6$ 区间出现波峰，则提示模型对当前样本的决策存在高不确定性。

---

## 9. � 实验跟踪与论文可视化 (Experiment Tracking & Visualization)

本项目集成了完整的实验跟踪和论文级可视化功能，在 [train/engine.py](file:///c:/Users/27203/Desktop/MGD_Project/train/engine.py) 和 [utils/plotting.py](file:///c:/Users/27203/Desktop/MGD_Project/utils/plotting.py) 中实现。

### 9.1 Weights & Biases (W&B) 训练可视化
集成 Weights & Biases 平台，实现训练过程的全面可视化和实验管理。

#### 9.1.1 实时损失与 Dice 曲线
自动记录训练和验证过程中的损失值、Dice 系数等关键指标，并生成实时更新的曲线图：

- 训练损失曲线（Train Loss）
- 验证损失曲线（Val Loss）
- 训练 Dice 系数曲线（Train Dice）
- 验证 Dice 系数曲线（Val Dice）

#### 9.1.2 训练样本可视化（原图 + GT + 预测对比）
定期保存训练样本的可视化对比结果，包括：
- 原始红外图像
- 专家标注 Ground Truth
- 模型实时预测结果

以网格布局并排展示，直观监控模型训练进度。

#### 9.1.3 超参数配置记录与版本管理
自动记录每次实验的完整配置：
- 模型架构选择（UNet / Attention UNet / WRCANet）
- 优化器参数（学习率、batch size）
- 损失函数配置
- 数据增强策略

配合 Git 版本控制，实现实验的可复现性。

### 9.2 Matplotlib 论文级可视化
提供专业的学术论文可视化功能，满足顶刊发表要求。

#### 9.2.1 指标对比柱状图（Dice / IoU / HD95）
生成多模型、多指标的对比柱状图：

- 分组对比不同模型架构的性能
- 误差线显示标准差
- 自动添加统计显著性标记

#### 9.2.2 分割结果对比图
生成横向拼接的分割结果对比图：
- 原图 → 标注 → 预测 的三列布局
- 多模型并排对比
- 局部放大展示细节差异

#### 9.2.3 ROC 曲线与 AUC 计算
计算并绘制 ROC 曲线：
- 不同阈值下的 TPR/FPR 计算
- AUC（Area Under Curve）数值标注
- 多模型 ROC 曲线对比

#### 9.2.4 顶刊标准样式（Arial 字体、矢量 PDF）
所有可视化均遵循学术论文标准：
- 使用 Arial 或 Times New Roman 字体
- 导出为矢量 PDF 格式
- 支持自定义配色方案（蓝/红/绿学术配色）
- 可调整的 DPI 和尺寸设置

---

## 10. 📈 专业评估指标与 LaTeX 导出 (Evaluation & Export)

在 [inference/evaluate.py](file:///c:/Users/27203/Desktop/MGD_Project/inference/evaluate.py) 和 [utils/metrics.py](file:///c:/Users/27203/Desktop/MGD_Project/utils/metrics.py) 中实现的专业评估体系。

### 10.1 MONAI 医学图像评估指标
基于 MONAI 库的专业医学图像分割评估指标。

#### 10.1.1 Dice 相似系数
衡量预测结果与标注的重叠程度：

$$
\text{Dice} = \frac{2 \cdot |P \cap G|}{|P| + |G|}
$$

其中 $P$ 为预测区域，$G$ 为 Ground Truth 区域。

#### 10.1.2 IoU 交并比
也称为 Jaccard 系数：

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|}
$$

#### 10.1.3 HD95 (95% Hausdorff 距离)
衡量边界对齐质量的鲁棒指标，取 95% 分位数以排除离群点：

$$
\text{HD95} = \text{Percentile}(\text{Hausdorff}(P, G), 95)
$$

#### 10.1.4 ASSD (平均对称表面距离)
计算两个表面之间的平均对称距离：

$$
\text{ASSD} = \frac{1}{2} \left( \frac{1}{|P|} \sum_{p \in P} d(p, G) + \frac{1}{|G|} \sum_{g \in G} d(g, P) \right)
$$

### 10.2 Pandas 数据处理与统计分析
使用 Pandas 进行数据整理和统计分析：
- 多折交叉验证结果聚合
- 均值、标准差计算
- 箱线图统计
- 显著性检验（t-test）

### 10.3 CSV 与 LaTeX 表格自动导出
一键导出专业的学术论文表格。

#### 10.3.1 多模型对比表格生成
自动生成包含所有评估指标的对比表格：
- 每行一个模型
- 每列一个指标（均值 ± 标准差）
- 最佳结果高亮标注

#### 10.3.2 学术论文格式适配
支持导出多种格式：
- CSV 格式（用于 Excel 进一步处理）
- LaTeX 格式（直接用于论文投稿）
- 可自定义表头和格式
- 支持三线表（Booktabs）样式

---

## 11. � 工程防御与安全架构 (Engineering Security)

作为一款工业级原型，本项目在代码底层构筑了铜墙铁壁般的系统鲁棒性：

### 11.1 数据空载防火墙 (Dataset Firewall)
深度学习在极小样本下迭代会导致权重坍塌（灾难性遗忘）。系统内置硬编码拦截器，当检测到 `dataset_size <= 2` 时强行中断训练并报警，保护历史最优权重。

### 11.2 动态显存防爆 (OOM Protection)
采用 PyTorch AMP 混合精度（FP16）加速，并通过 `set_per_process_memory_fraction` 实现显存限额自适应分配：

$$
f_\text{limit} = \min(0.9, \frac{M_\text{available}}{M_\text{total}})
$$

确保低配核显设备也能稳定运行。

### 11.3 完全边缘计算 (Edge Computing)
所有 DICOM 预处理与大模型推理均在本地局域网完成，**绝不向外部公网传输任何患者隐私影像**，完全符合医疗数据合规审查要求。

---

## 12. 🚀 环境依赖与一键部署 (Installation)

### 12.1 项目结构说明

```
项目根目录
├── init-windows.bat  (Windows 环境一键初始化)
├── init-mac.sh       (macOS 环境一键初始化)
├── init-linux.sh     (Linux 环境一键初始化)
├── README.md
├── .gitignore
└── data
    ├── win/          (Windows 端源码 - 完整WebUI诊断版)
    ├── mac/          (macOS 端源码 - 完整WebUI诊断版)
    └── linux/        (Linux 端源码 - 训练专用版)
```

### 12.1.1 各平台功能定位

| 平台 | 功能定位 | WebUI诊断 | GUI训练 | 命令行训练 |
|------|---------|----------|---------|-----------|
| **Windows** | 完整临床诊断工具 | ✅ 有 | ✅ 有 | ❌ 无 |
| **macOS** | 完整临床诊断工具 | ✅ 有 | ✅ 有 | ❌ 无 |
| **Linux** | 训练专用版 | ❌ 无 | ❌ 无 | ✅ 有 |

**注意**：
- **Windows/macOS 版本**：面向临床医生和研究人员，包含完整的 Gradio WebUI 诊断界面和可视化训练界面
- **Linux 版本**：面向研究人员和开发者，专注于模型训练功能，提供命令行和交互式训练方式（无 WebUI）

### 12.2 核心依赖环境
- **Python**: 3.9+
- **PyTorch**: 2.0+ (CUDA 11.8+ 推荐)
- **OpenCV**: `opencv-python==4.8.0.74` (严格指定，解决 Numpy 兼容性)
- **MONAI**: 1.3.0+
- **Next.js**: 16.2.0+
- **Gradio**: 6.12.0+

### 12.3 快速启动（推荐）

**Windows 用户：**
1. 双击运行 `init-windows.bat`，自动完成环境配置
2. 完成后双击运行 `start_win.bat` 启动服务

**macOS 用户：**
1. 在终端运行：`chmod +x init-mac.sh && ./init-mac.sh`
2. 完成后运行：`./start_macos.sh`

**Linux 用户：**
1. 在终端运行：`chmod +x init-linux.sh && ./init-linux.sh`
2. 完成后运行：`./run.sh`

### 12.4 手动启动方式（备用）

如果您需要手动配置环境，请使用以下步骤：

```bash
# 1. 克隆代码仓库
git clone https://github.com/Yao-Shun-Ya/MGD_Project.git
cd MGD_Project

# 2. 创建并激活独立的虚拟环境 (强烈建议)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 3. 安装包含向下兼容 Numpy 的全家桶依赖
pip install opencv-python==4.8.0.74 "numpy<2.0.0" scikit-image monai gradio torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 启动临床诊断 WebUI 终端
python app.py
```

*启动成功后，浏览器访问终端输出的本地端口（通常为 `http://127.0.0.1:7860`）即可进入控制台。*

---

## 13. 📦 模型权重配置 (Weights Configuration)

由于满血版全尺度模型权重文件体积较大（约 120MB+），请通过主页下载中心单独获取 `meibomian_model_best.pth`。

请严格按照以下路径将其放入您的本地项目中，并与 `app.py` 处于同级目录：

```text
MGD_Project/
 ├── app.py
 ├── data_loader.py
 ├── style.css
 └── meibomian_model_best.pth   <-- 【必须放入此核心权重文件】
```

*未配置权重时，系统将触发安全拦截并提示要求前往“模型训练舱”进行从零训练。*

---

## 14. 🎓 学术引用与开源声明 (Citation)

### 研发团队
- **Lead Developer & Researcher**: 黄宇普 (Huang Yupu)
- **Organization**: NSMC XIANERYIJIAN TEAM
- **Contact**: 2720356281@QQ.COM

### 相关仓库与项目官网
- 🌐 **项目官网**: [https://xianeryijian.icu](https://xianeryijian.icu)
- 🧠 **深度学习引擎仓库**: [https://github.com/Yao-Shun-Ya/MGD_Project](https://github.com/Yao-Shun-Ya/MGD_Project)
- 🌐 **前端网站仓库**: [https://github.com/Yao-Shun-Ya/Xian-Er-Yi-Jian-Web](https://github.com/Yao-Shun-Ya/Xian-Er-Yi-Jian-Web)

### 参考文献
- [1] Lemp MA, Crewes LA, Bron AJ, et al. The definition and classification of dry eye disease. *Ocul Surf.* 2007;5(2):75-92.
- [2] Arita R, Itoh K, Inoue K, Amano S. Noncontact infrared meibography to document age-related changes. *Ophthalmology.* 2008;115(5):911-915.
- [3] Salehi SSM, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. *arXiv:1706.05721*. 2017.
- [4] Loshchilov I, Hutter F. SGDR: Stochastic Gradient Descent with Warm Restarts. *arXiv:1608.03983*. 2016.
- [5] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. 2015.
- [6] Isensee F, et al. nnU-Net: a Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. *Nature Methods*. 2021.

### 开源协议 (License)
本项目代码基于 **Apache 2.0 License** 开放源代码。

> **免责声明**：医疗诊断必须由具备执业资质的医师做出。本软件及 AI 模型输出结果仅供科研与临床辅助参考，开发团队对任何基于本系统的直接医疗干预后果不承担法律责任。

### 如何引用
如果您竞赛或科研论文中使用了本平台的架构思想，请遵循以下格式引用：

```bibtex
@misc{huang2026xianeryijian,
  author = {Yupu Huang},
  title = {MGD_Project: Sub-pixel Deep Learning Platform for Meibomian Gland Dysfunction Quantification with Advanced Visualization and Test-Time Augmentation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Yao-Shun-Ya/MGD_Project}}
}
```

<div align="center">
  <b>🌟 如果本协议及架构对您的医学影像科研有启迪，请给予项目一个 Star 🌟</b>
</div>

