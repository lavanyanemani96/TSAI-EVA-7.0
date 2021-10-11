# Session 2.5 Pytorch 101 

## Goal
Create a network that takes two inputs; handwritten digit image and a random single digit number and outputs the number that's on the image and the sum of it and the random number. 

## Data representation and generattion
We use datasets.MNIST to get MNIST handwritten digits dataset. This is a set of 60000 training images and 10000 test images. For the random number we use numpy.random.randint to generate a number between 0 and 9. class mnist_data is used to create this custom dataset using a custom getitem method. 

## Combining the two inputs 
The network concatenates the two inputs after the image reaches a bottleneck of size 10 in a dimension, since the 1-hot encoding of the random number is also of size 10. Then fully connected layers are used to do the addition between the numbers. 

## Loss function 
We use crossentropy loss since it is useful when training a classification problem with C classes. Here for the images the 10 classes are the digits (0, 1, .. , 9) and similarly for the sum the 19 classes are the possible sums (0, 1, .. , 18).


## Evaluation 
The results are evaluated on a test set that isn't used to update the weights of the network. This is done for each epoch using total crossentropy loss from image label and sum label and also shows the accuracy (% of correct labels) at each epoch.

## Results 
After training for 5 epochs we get 99% accuracy on image labels and 97% accuracy on the predicted sum. 

## Training logs:
Train loss=2.122750759124756 batch_id=599: 100%|██████████| 600/600 [00:45<00:00, 13.17it/s]

Test set: Average loss: 0.0209, Accuracy image labels: 9542/10000 (95%), Accuracy sum labels: 2378/10000 (24%)
Train loss=1.3610708713531494 batch_id=599: 100%|██████████| 600/600 [00:45<00:00, 13.12it/s]

Test set: Average loss: 0.0131, Accuracy image labels: 9851/10000 (99%), Accuracy sum labels: 5189/10000 (52%)
Train loss=0.5874066352844238 batch_id=599: 100%|██████████| 600/600 [00:45<00:00, 13.09it/s]

Test set: Average loss: 0.0069, Accuracy image labels: 9903/10000 (99%), Accuracy sum labels: 8287/10000 (83%)
Train loss=0.2033192366361618 batch_id=599: 100%|██████████| 600/600 [00:45<00:00, 13.09it/s]

Test set: Average loss: 0.0024, Accuracy image labels: 9913/10000 (99%), Accuracy sum labels: 9517/10000 (95%)
Train loss=0.10257242619991302 batch_id=599: 100%|██████████| 600/600 [00:45<00:00, 13.08it/s]

Test set: Average loss: 0.0012, Accuracy image labels: 9936/10000 (99%), Accuracy sum labels: 9745/10000 (97%)

