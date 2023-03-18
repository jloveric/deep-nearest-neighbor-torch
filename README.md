# Dynamic and Deep Nearest Neighbors in Pytorch
How far can nearest neighbors go? Experiments to see

This algorithm does weighted (interpolated) nearest neighbor approximation, building the
approximation in batch mode.  Each batch adds the "centers" or "neighbors" that are not approximated correctly from the existing model, so instead of using the entire dataset as the list of neighbors, only uses a subset. The entire thing runs on the GPU using pytorch.  During evaluation, instead of taking the k-nearest neighbor a weighted average of all neighbors
is used to compute the output. The weighting is a function of the inverse distance or a cone of influence, which is defined as a factor of the distance to the nearest neighbor.

# Invariant MNIST
To run
```
python examples/invariant_image_classification.py data=mnist exponent=-4 kernel_type=cosine
```
```
Current working directory : /mnt/1000gb/deep-nearest-neighbor-py/outputs/2023-03-18/15-10-57
100%|████████████████████████████████████████████████████████████████████████████████▉| 937/938 [00:20<00:00, 46.29it/s, neighbors=2911]
Epoch_loop time 20.243689922848716
Network neighbors 2912
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 61.70it/s]
Epoch_loop time 15.202289984095842
train_result Results(error=0.007466666666666667, accuracy=0.9925333333333334, incorrect=448, total=60000)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 67.81it/s]
Epoch_loop time 2.3158813391346484
test_result Results(error=0.0312, accuracy=0.9688, incorrect=312, total=10000)
neighbors in model 2912
```
# Invariant CIFAR100
Not so good
```
python examples/invariant_image_classification.py data=cifar100 exponent=-4 kernel_type=cosine
```
```
Current working directory : /mnt/1000gb/deep-nearest-neighbor-py/outputs/2023-03-18/15-17-15
100%|███████████████████████████████████████████████████████████████████████████████▉| 781/782 [01:17<00:00, 10.11it/s, neighbors=42363]
Epoch_loop time 77.27613210305572
Network neighbors 42378
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:44<00:00, 17.58it/s]
Epoch_loop time 44.47933694208041
train_result Results(error=0.0287, accuracy=0.9713, incorrect=1435, total=50000)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:08<00:00, 18.48it/s]
Epoch_loop time 8.494573957985267
test_result Results(error=0.8207, accuracy=0.1793, incorrect=8207, total=10000)
neighbors in model 42378
```
# Invariant CIFAR10
Better than CIFAR100
```
python examples/invariant_image_classification.py data=cifar10 exponent=-4 kernel_type=cosine
```
```
Current working directory : /mnt/1000gb/deep-nearest-neighbor-py/outputs/2023-03-18/15-23-55
100%|███████████████████████████████████████████████████████████████████████████████▉| 781/782 [00:24<00:00, 32.18it/s, neighbors=30859]
Epoch_loop time 24.273869331926107
Network neighbors 30867
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:17<00:00, 45.65it/s]
Epoch_loop time 17.131763230077922
train_result Results(error=0.05982, accuracy=0.94018, incorrect=2991, total=50000)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 44.45it/s]
Epoch_loop time 3.532990956911817
test_result Results(error=0.5865, accuracy=0.4135, incorrect=5865, total=10000)
neighbors in model 30867
```


# Image approximation using weighted neighbor approximation
The example below is approximately 1/10 data compression of the original image.
The left image shows the locations of the "centers", the right image shows the
approximation of the original using these centers. Training completes in a 3
seconds with large batch=256 on a NVidia 2080 and uses about 16,000 centers of the
196,608 centers available in the data set.
```
python examples/image_interpolation.py tolerance=0.2 target_accuracy=0.8
```
![Image Approximation](results/NearestNeighborApproximation.png)

# Language generation
Run
```
python examples/language_generation.py 
```
testing
```
python examples/language_generation.py train=False directory=/mnt/1000gb/deep-nearest-neighbor-py/outputs/2023-03-12/12-56-39 text_prompt="What is this about"
```