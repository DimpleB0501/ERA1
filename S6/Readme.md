# Assignment Part 2
Q. WRITE The given code AGAIN SUCH THAT IT ACHIEVES<br/>
- 99.4% validation accuracy<br/>
- Less than 20k Parameters<br/>
- Less than 20 Epochs<br/>
- Have used BatchNormalization, Dropout, a Fully connected layer, have used GAP. <br/>
##### Solution
I started with modification of the [LENET](https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/) architecture. <br/>
My network consists of 3 convolution layers, 2 fully connected layers, average pooling layers, batch normalization, dropout  and GAP layer (which I have added using `self.gap = torch.nn.AdaptiveAvgPool2d((1,1))` function).<br/>
The total number of parameters in my network are **16,994 (< 20k)** <br/>
Following figure displays output from **torchsummary** <br/>
![layers](images/summ.png)

I received an accuracy of 99.4% at the **16th** epoch.
![acc](images/acc.png)
