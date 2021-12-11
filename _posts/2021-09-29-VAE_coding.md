---
title: Math AI - VAE Coding 
date: 2021-09-29 23:10:08
categories:
- AI
tags: [ML, VAE, Autoencoder, Variational, EM]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>

## Main Reference

* [@kingmaIntroductionVariational2019] : excellent reference

* [@kingmaAutoEncodingVariational2014]

* [@roccaUnderstandingVariational2021]

## VAE Recap

Recap VAE spirit: marginal likelihood = ELBO + gap => focus on ELBO only!

$$
\begin{aligned}\log p_{\boldsymbol{\theta}}(\mathbf{x}) &=\underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=\mathcal{L}_{\theta,\phi}{(\boldsymbol{x}})\,\text{, ELBO}}+\underbrace{\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=D_{K L}\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right)}\end{aligned}
$$

$$
\begin{aligned}\underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=\mathcal{L}_{\theta,\phi}{(\boldsymbol{x}})\,\text{, ELBO}} &= \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}\left[\log p_{\theta}(\mathbf{x} | \mathbf{z})\right] - D_{K L}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \|\,p(\mathbf{z})\right) \\&= (-1) \times \text{VAE Loss Function}\end{aligned}
$$

With the loss function, We can start training.

* Gradient
* Some term are samples (1), some has analytical form (2) (see appendix A)

(1) Naive Monte Carlo gradient estimator

 $\nabla_{\phi} E_{q_{\phi}(\mathbf{z})}[f(\mathbf{z})] = E_{q_{\phi}(\mathbf{z})}\left[f(\mathbf{z}) \nabla_{q_{\phi}(\mathbf{z})} \log q_{\phi}(\mathbf{z})\right] \simeq \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{z}) \nabla_{q_{\phi}\left(\mathbf{z}^{(l)}\right)} \log q_{\phi}\left(\mathbf{z}^{(l)}\right)$

where $\mathbf{z}^{(l)} \sim q_{\phi}\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)$.

This gradient estimator exhibits exhibits very high variance (see e.g. [BJP12])

### SGVB estimator and AEVB algorithm

這節討論實際的 estimator of approximate posterior in the form of $q_\phi(\mathbf{z}\mid \mathbf{x})$. 注意也可以適用於 $q_\phi(\mathbf{z})$.  

Under certain mild conditions outlined in section 2.4 for a chosen approximate posterior $q_\phi(\mathbf{z}\mid \mathbf{x})$ we can reparametrize the random variable $\tilde{\mathbf{z}} \sim q_\phi(\mathbf{z}\mid \mathbf{x})$ using a differentiable transformation $g_{\phi}(\epsilon, x)$ of an (auxiliary) noise variable :

$$E_{q_{\phi}\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)}[f(\mathbf{z})]=E_{p(\epsilon)}\left[f\left(g_{\phi}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \simeq \frac{1}{L} \sum_{l=1}^{L} f\left(g_{\phi}\left(\boldsymbol{\epsilon}^{(l)}, \mathbf{x}^{(i)}\right)\right) \quad$ where $\quad \boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})$$

We apply this technique to the variational lower bound (eq. (2)), yielding our generic Stochastic Gradient Variational Bayes (SGVB) estimator $\widetilde{\mathcal{L}}^{A}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right) \simeq \mathcal{L}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right)$ :

$$
\widetilde{\mathcal{L}}^{A}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right)=\frac{1}{L} \sum_{l=1}^{L} \log p_{\boldsymbol{\theta}}\left(\mathbf{x}^{(i)}, \mathbf{z}^{(i, l)}\right)-\log q_{\phi}\left(\mathbf{z}^{(i, l)} \mid \mathbf{x}^{(i)}\right)
$$

where $\quad \mathbf{z}^{(i, l)}=g_{\phi}\left(\boldsymbol{\epsilon}^{(i, l)}, \mathbf{x}^{(i)}\right) \quad$ and $\quad \boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})$

##### Algorithm 1:  Minibatch version of Auto-Encoding Variational Bayes (AEVB) algorithm.  We set M=100 and L=1

$\theta, \phi$ : Initialize parameters

Repeat

* $X^M$ Random minibatch of M datapoints (drawn from full dataset)

* $\boldsymbol{\epsilon}$  Random samples from noise distribution $p(\boldsymbol{\epsilon})$

* $\mathbf{g}$  gradients of minibatch estimator

* $\theta, \phi$  Update parameters using gradients $\mathbf{g}$

??? SGVB estimator $\widetilde{\mathcal{L}}^{B}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right) \simeq \mathcal{L}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right)$, corresponding to eq. (3), which typically has less variance than the generic estimator:

$$
\widetilde{\mathcal{L}}^{B}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right)=-D_{K L}\left(q_{\boldsymbol{\phi}}\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right) \| p_{\boldsymbol{\theta}}(\mathbf{z})\right)+\frac{1}{L} \sum_{l=1}^{L}\left(\log p_{\boldsymbol{\theta}}\left(\mathbf{x}^{(i)} \mid \mathbf{z}^{(i, l)}\right)\right)
$$

where $\quad \mathbf{z}^{(i, l)}=g_{\phi}\left(\boldsymbol{\epsilon}^{(i, l)}, \mathbf{x}^{(i)}\right) \quad$ and $\quad \boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})$

Given multiple datapoints from the dataset $X$ with N datapoints, we can

$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{X}) \simeq \widetilde{\mathcal{L}}^{M}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{X}^{M}\right)=\frac{N}{M} \sum_{i=1}^{M} \widetilde{\mathcal{L}}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right)
$$

#### Example: Variational Auto-Encoder, assuming Gaussian

$$
\mathcal{L}\left(\boldsymbol{\theta}, \boldsymbol{\phi} ; \mathbf{x}^{(i)}\right) \simeq \frac{1}{2} \sum_{j=1}^{J}\left(1+\log \left(\left(\sigma_{j}^{(i)}\right)^{2}\right)-\left(\mu_{j}^{(i)}\right)^{2}-\left(\sigma_{j}^{(i)}\right)^{2}\right)+\frac{1}{L} \sum_{l=1}^{L} \log p_{\theta}\left(\mathbf{x}^{(i)} \mid \mathbf{z}^{(i, l)}\right)
$$

where $\quad \mathbf{z}^{(i, l)}=\boldsymbol{\mu}^{(i)}+\boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\epsilon}^{(l)} \quad$ and $\quad \boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(0, \mathbf{I})$

### VAE Encoder-Decoder Structure

From [@roccaUnderstandingVariational2021],一個是 encoder NN, 如下式 $(g^*, h^*)$

$$
\begin{aligned}
\left(g^{*}, h^{*}\right) &=\underset{(g, h) \in G \times H}{\arg \min } K L\left(q_{x}(z), p(z \mid x)\right) \\
&=\underset{(g, h) \in G \times H}{\arg \max }\left(\mathbb{E}_{z \sim q_{x}}\left(-\frac{\|x-f(z)\|^{2}}{2 c}\right)-D_{K L}\left(q_{x}(z), p(z)\right)\right)
\end{aligned}
$$

另一個是 decoder NN, 如下式 $f^*$

$$
\begin{aligned}
f^{*} &=\underset{f \in F}{\arg \max } \mathbb{E}_{z \sim q_{x}^{*}}(\log p(x \mid z)) \\
&=\underset{f \in F}{\arg \max } \mathbb{E}_{z \sim q_{x}^{*}}\left(-\frac{\|x-f(z)\|^{2}}{2 c}\right)
\end{aligned}
$$

Gathering all the pieces together, we are looking for optimal $\mathrm{f}^{*}, \mathrm{~g}$* and $\mathrm{h}^{*}$ such that

$$
\left(f^{*}, g^{*}, h^{*}\right)=\underset{(f, g, h) \in F \times G \times H}{\arg \max }\left(\mathbb{E}_{z \sim q_{x}}\left(-\frac{\|x-f(z)\|^{2}}{2 c}\right)-D_{K L}\left(q_{x}(z), p(z)\right)\right)
$$

等價於 minimize VAE loss function

$$
\begin{aligned}
\text {VAE loss }&=C\|x-\hat{x}\|^{2}+D_{KL}\left(N\left(\mu_{x}, \sigma_{x}\right), N(0, I)\right)\\
&=C\|x-f(z)\|^{2}+D_{KL}(N(g(x), h(x)), N(0, l))
\end{aligned}
$$

第一項是 reconstruction loss, 第二項是 regularization loss.  第一項從 sampling 得到。第二項有 analytical form, 見 Appendix A.

In practice, g and h are not defined by two completely independent NN but share a part of their architecutre and theier weights so that

$\mathbf{g}(x) = \mathbf{g}_2(\mathbf{g}_1(x)) \quad  \mathbf{h}(x) = \mathbf{h}_2(\mathbf{h}_1(x)) \quad \mathbf{g}_1(x) = \mathbf{h}_1(x)$

<img src="/media/img-2021-10-02-20-25-39.png" style="zoom:67%;" />

<img src="/media/img-2021-10-02-21-06-37.png" style="zoom:67%;" />



#### Binary Image Approximation Using Bernoullie Distribution

如果 image 是黑白二值 (binary black and white), 可以用 Bernoulli distributionm.  Reconstruction loss 可以改用 binary cross entropy loss, 而不是 上面的 MSE loss.[^1]

[^1]: Reference: https://spaces.ac.cn/archives/5343 

$$
p(\xi)=\left\{\begin{array}{l}
\rho, \xi=1 \\
1-\rho, \xi=0
\end{array}\right.
$$
Bernoulli distribution 適用於多個二值向量的情况，比如 $x$ 是 binary image (mnist可以看成這種例子，雖然是 grey value 而不是 binary value)
$$
q(x \mid z)=\prod_{k=1}^{D}\left(\rho_{(k)}(z)\right)^{x_{(k)}}\left(1-\rho_{(k)}(z)\right)^{1-x_{(k)}}
$$
$$
-\ln q(x \mid z)=\sum_{k=1}^{D}\left[-x_{(k)} \ln \rho_{(k)}(z)-\left(1-x_{(k)}\right) \ln \left(1-\rho_{(k)}(z)\right)\right]
$$

這表明 $\rho(z)$ 要把 output 壓縮在 0~1 (例如用 sigmoind activation), 然後用 BCE 做為 reconstruction loss function,



#### 以下是 VAE PyTorch code example for MNIST

##### MNIST dataset

* MNIST image: size 28x28=784 pixels of grey value between 0 and 1.  0: 白；1：黑。0.1-0.9 代表不同的灰階，如下圖。
* MNIST datset: 60K for training;  10K for testing.  Total 70K.

<img src="/media/img-2021-10-03-09-27-59.png" style="zoom:67%;" />

##### VAE Model

* VAE **encoder** first uses FC network (fc1: 784->400) + ReLU, 等價上圖的 $\mathbf{h}_1 = \mathbf{g}_1$
* 再接上兩個 FCs (fc21=$\mathbf{g}_2$, fc22=$\mathbf{h}_2$, 400->20) 產生 mean,mu, and log of variance, logvar of 20 dimensions.  **注意這二個 FCs 沒有串接 ReLU, 因爲 mean and logvar 可正可負。**
* 基於 reparameterization trick:  $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} $ (20-dimension)
* VAE **decoder** 先是 FC network (fc3, 20->400) + ReLU
* 再串一個 FC network (fc4, 400->784=28x28) + sigmoid 保證介於 0~1 (to match mnist image grey level).  也就是 $\mathbf{f}$ = fc3+ReLU+fc4+sigmoid
* Forward path 包含 encode, reparameterize, decode.

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
```

##### VAE Loss function and optimizer

* 注意這裡VAE loss function 完全不用 label, i.e. 0, 1, ..., 9.  可以說是 self-supervised learning. 
* BCE 是 binary cross-entropy, 代表 reconstruction loss. 注意雖然稱爲 binary cross-entropy, label 可以是 0-1 的 value, 因爲 mnist 的 image 是 grey level 而非 binary value.  爲什麽是 reduction = sum 而非 mean?
* KLD 是 KL divergence, 是 regularization term.  在 Gaussian assumption 有 analytical form.

```python
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix A from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

##### 整合 training code

* Training dataset (60K) 由 train_loader 載入。Mini-batch size 可由 command line 指定, default = 128.
* model(data) 完成 forward, 傳回 reconstructed image, mu, logvar 用於 loss computation with batch_size=128.  就是**每張 image 的 loss** 纍積 128 張。
* 接著每個 mini-batch 計算 backward and use Adam optimizer to update weights.  不過爲了避免雜亂，只有 log_interval (default=10) 才 print 一次 log, default = 128x10 = 1280.
* 每個 epoch print average training loss (default 10 epoches).

```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
```

##### 結果

* 每一次 log 是 128x10=1280, 大於 2% of 60K dataset per epoch.
* Epoch 1 average loss 很大：164.  到了 Epoch 10 average loss: 106.  基本已經 saturated. 這個 loss 包含 BCE and KLD.  
  * Total loss: Epoch 1 ~ 164;  Epoch 10 ~ 106.
  * KLD loss:   Epoch 1 ~  14;   Epoch 10 ~ 25.
  * BCE loss:   Epoch 1 ~ 150;  Epoch 10 ~ 81.
* BCE loss 就是一般 autoencoder loss 隨著 epoch 增加變小，但 KLD loss 變大，同時 regularize BCE loss saturate.  

```
Train Epoch: 1 [0/60000 (0%)]           Loss: 550.513977
Train Epoch: 1 [1280/60000 (2%)]        Loss: 310.610535
.... omit
Train Epoch: 1 [57600/60000 (96%)]      Loss: 129.696487
Train Epoch: 1 [58880/60000 (98%)]      Loss: 132.375336
====> Epoch: 1 Average loss: 164.4209

... Epoch 2 to 9, TL;DP

Train Epoch: 10 [0/60000 (0%)]          Loss: 105.353363
Train Epoch: 10 [1280/60000 (2%)]       Loss: 103.786560
... omit
Train Epoch: 10 [57600/60000 (96%)]     Loss: 107.218582
Train Epoch: 10 [58880/60000 (98%)]     Loss: 105.427353
====> Epoch: 10 Average loss: 106.1371

```

下圖左上和左下對應 epoch 1 的 reconstructed images 和 random generated images.
下圖右上和右下對應 epoch 10 的 reconstructed images 和 random generated images. 都是 20-dimension.

<img src="/media/img-2021-10-03-12-20-16.png" style="zoom:67%;" />

#### Appendix A - Solution of Gaussian Distribution of $D_{K L}(q_\phi(\mathbf{z})\|p_{\theta}(\mathbf{z}))$

$$
\begin{aligned}
\int q_{\boldsymbol{\theta}}(\mathbf{z}) \log p(\mathbf{z}) d \mathbf{z} &=\int \mathcal{N}\left(\mathbf{z} ; \boldsymbol{\mu}, \boldsymbol{\sigma}^{2}\right) \log \mathcal{N}(\mathbf{z} ; \mathbf{0}, \mathbf{I}) d \mathbf{z} \\
&=-\frac{J}{2} \log (2 \pi)-\frac{1}{2} \sum_{j=1}^{J}\left(\mu_{j}^{2}+\sigma_{j}^{2}\right)
\end{aligned}
$$

And:

$$
\begin{aligned}
\int q_{\boldsymbol{\theta}}(\mathbf{z}) \log q_{\boldsymbol{\theta}}(\mathbf{z}) d \mathbf{z} &=\int \mathcal{N}\left(\mathbf{z} ; \boldsymbol{\mu}, \boldsymbol{\sigma}^{2}\right) \log \mathcal{N}\left(\mathbf{z} ; \boldsymbol{\mu}, \boldsymbol{\sigma}^{2}\right) d \mathbf{z} \\
&=-\frac{J}{2} \log (2 \pi)-\frac{1}{2} \sum_{j=1}^{J}\left(1+\log \sigma_{j}^{2}\right)
\end{aligned}
$$

Therefore:

$$
\begin{aligned}
-D_{K L}\left(\left(q_{\phi}(\mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{z})\right)\right.&=\int q_{\boldsymbol{\theta}}(\mathbf{z})\left(\log p_{\boldsymbol{\theta}}(\mathbf{z})-\log q_{\theta}(\mathbf{z})\right) d \mathbf{z} \\
&=\frac{1}{2} \sum_{j=1}^{J}\left(1+\log \left(\left(\sigma_{j}\right)^{2}\right)-\left(\mu_{j}\right)^{2}-\left(\sigma_{j}\right)^{2}\right)
\end{aligned}
$$
When using a recognition model $q_{\phi}(z|x)$ then $\mu$ and s.d. $\sigma$ are simply functions of $x$ and the variational parameters $\phi$, as exemplified in the text.
