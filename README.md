# Enhancing Cancer Detection with StyleGAN-2 ADA
We use StyleGAN-2 with Adaptive Disriminator Augmentation (ADA) to generate synthetic samples of a small [chest CT-scan dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code?datasetId=839140&sortBy=voteCount). 

ADA allows breakthrough performance when training GANs on small amounts of data. This technique was introduced by NVIDIA in the NeurIPS 2020 paper ["Training Generative Adversarial Networks with Limited Data"](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/ada-paper.pdf). Being able to accurately augment data within domains where data itself is limited, private, or difficult to obtain is a very important task.

We test the efficacy of these synthetic samples by mixing them into the training process of a CNN, where we classify normal versus cancerous CT-scans. We tested benchmark performance on a model with only the original data and compared it to a model trained on an augmented mixed dataset. We observed accuracy improvements on the test set through the added use of GAN-generated synthetic samples in the CNN training.

For more information, please refer to the following article on [Medium]().
