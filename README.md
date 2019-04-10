# Adversial_AutoEncoder using MNIST data
Computation Graph 
![Generative_Model](https://user-images.githubusercontent.com/21220616/55679013-23bea400-5921-11e9-8244-0aff3ec7a8e7.png)

## As Mentioned in paper ![Loss](https://user-images.githubusercontent.com/21220616/55679043-a5163680-5921-11e9-9a61-49e44b863087.png)

## Parameter of Model
Prior distribution are taken as : Standard Bivariate Noramal Distribution mean [0,0] and Covariance [[1,0],[0,1]]
Latent dimension of Input image is taken as 2
Learning rate for Updating Encoder in regularization training are taken small or we can equivalently weight the loss with smaller number in comparision to loss in reconstruction

## Encoder architecture are as:
input (batch_size X 784) ----fc---> (784 X 400) --fc-->(400 X 100) ---fc-->(100 X 2) 

## Decoder architecture are as:

input(batch_size X 2) ---fc---> (2,100) ---fc---> (100 X400) ---fc--> (400 X784)

## Discriminator architecture are as:

input(batch_size X 2) ---fc--->(2,10) ---fc---> (10,10) ---fc---> (10,2) (softmax layer)

