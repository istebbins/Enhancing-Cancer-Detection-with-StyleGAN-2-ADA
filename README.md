# Enhancing Cancer Detection with StyleGAN-2 ADA
We use StyleGAN-2 with Adaptive Disriminator Augmentation (ADA) to generate synthetic samples of a small [chest CT-scan dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code?datasetId=839140&sortBy=voteCount). 

ADA allows breakthrough performance when training GANs on small amounts of data. This technique was introduced by NVIDIA in the NeurIPS 2020 paper ["Training Generative Adversarial Networks with Limited Data"](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/ada-paper.pdf). Being able to accurately augment data within domains where data itself is limited, private, or difficult to obtain is a very important task.

We test the efficacy of these synthetic samples by mixing them into the training process of a CNN, where we classify normal versus cancerous CT-scans. We tested benchmark performance on a model with only the original data and compared it to a model trained on an augmented mixed dataset. We observed accuracy improvements on the test set through the added use of GAN-generated synthetic samples in the CNN training.

For more information, please refer to the following article on [Medium](https://medium.com/@ianstebbs/aee55ef99c5b).


# Convolutional Neural Network for Lungs Classification

## Important: 
Here is a [link](https://www.kaggle.com/datasets/benjaminmaizes/formatted-and-augmented-chest-ct-scan-images) to our version of the dataset. Please use this when training the CNN. 

This README provides instructions on how to use the Convolutional Neural Network (CNN) provided in the accompanying notebook to classify images of squamous cell or normal cell lungs. The model is designed to work with 224 x 224 images and requires users to set up their own image directory with a desired proportion of synthetic and real data. Follow the steps below to prepare and run the model:

## Prerequisites

Before using the CNN, ensure you have the following:

1. Python installed on your machine.
2. Necessary Python packages installed, including TensorFlow and Keras.

## Setup Instructions

1. **Create Your Image Directory:**
   - Organize your dataset into three directories: one for training data, one for validation data, and one for testing data.
   - Download the dataset from the above link. 
   - Ensure that the images in each directory are organized by class (e.g., 'squamous_cell' and 'normal_cell').
   - An ideal filepath looks like this:
     - Images 
         - Train
            - Normal
            - Squamous
         - Test
            - Normal
            - Squamous
         - Validate
            - Normal
            - Squamous
    - You can fille these directories with any proportion of real and synthetic data from the dataset you choose. 

2. **Specify Directories in the Notebook:**
   - Open the notebook and locate the section where directories are defined.
   - Set the paths for the training, validation, and testing datasets in the `train_dir`, `val_dir`, and `test_dir` variables, respectively.
   - If you use the above setup it might look like 
      - train_dir = "path/to/Images/Train"
      - test_dir = "path/to/Images/Test"
      - val_dir = "path/to/Images/Validate"
   

3. **Adjust ImageDataGenerator:**
   - The model uses `ImageDataGenerator` for data augmentation. You can customize the data augmentation parameters based on your requirements.

4. **Model Configuration:**
   - The CNN uses the ResNet50 architecture. Fine-tuning is applied to train the last two layers for improved accuracy.
   - Adjust the model architecture, including the number of layers, units, and regularization parameters based on your needs.

5. **Compile the Model:**
   - Configure the model by setting the learning rate, loss function, and metrics in the `model.compile` section.

6. **Training:**
   - Specify the batch size and the number of epochs for training in the `batch_size` and `epochs` variables.
   - Adjust the `patience` and `stop_patience` variables for early stopping and learning rate adjustments during training.

7. **Callbacks:**
   - ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks are implemented. Adjust their parameters if needed.

8. **Training the Model:**
   - Run the training section of the notebook. The model will train on your specified dataset, and the training progress will be saved.

9. **Save the Model:**
   - Modify the `percentage` variable based on the percentage of synthetic data used.
   - The trained model will be saved to a specified location. Ensure that the path is correct and that you have write permissions.

10. **Evaluate and Use:**
    - Evaluate the model on your test dataset and use it for inference as needed.

11. **Example:**
    - There is a saved model you can load in saved_100per_model with accuracy: 0.9861 on the test data if you want a pre-trained model. This model is located online with the dataset on kaggle. 

Remember to check the notebook for comments and documentation provided by the original author for additional insights and customization options.

Feel free to reach out if you have any questions or encounter issues during the setup process. Good luck with your lung classification project!
