# Heterogeneous image classification

</small>
## Image Classification Results for Cifar10 CUDA

| File                                   | File Count | Predicted Image | Total Time (ms) | Total Time Taken Up to Now (ms) |
|----------------------------------------|------------|-----------------|-----------------|---------------------------------|
| flattened_airplane_airplane_100.txt    | 1          | airplanes       | 0.205           | 0.205                           |
| flattened_airplane_airplane_101.txt    | 2          | airplanes       | 0.384           | 0.589                           |
| flattened_airplane_airplane_102.txt    | 3          | airplanes       | 0.365           | 0.954                           |
| flattened_airplane_airplane_103.txt    | 4          | airplanes       | 0.382           | 1.336                           |
| flattened_airplane_airplane_104.txt    | 5          | airplanes       | 0.732           | 2.068                           |
| flattened_airplane_airplane_105.txt    | 6          | airplanes       | 0.706           | 2.774                           |
| *** ... ***                            | ***        | ***             | ***             | ***                             |
| flattened_truck_truck_9.txt            | 1154       | trucks          | 0.37            | 792.003                         |

Total CUDA execution time: 792.003 milliseconds

## Image Classification Results for Cifar10 C++ and OpenMP

| File                                   | File Count | Predicted Image | Total Time (ms) | Total Time Taken Up to Now (ms) |
|----------------------------------------|------------|-----------------|-----------------|---------------------------------|
| flattened_airplane_airplane_100.txt    | 1          | airplanes       | 197.695         | 197.695                         |
| flattened_airplane_airplane_101.txt    | 2          | airplanes       | 52.109          | 249.804                         |
| flattened_airplane_airplane_102.txt    | 3          | airplanes       | 213.931         | 463.735                         |
| flattened_airplane_airplane_103.txt    | 4          | airplanes       | 63.163          | 526.898                         |
| flattened_airplane_airplane_104.txt    | 5          | airplanes       | 75.58           | 602.478                         |
| flattened_airplane_airplane_105.txt    | 6          | airplanes       | 195.106         | 797.584                         |
| *** ... ***                            | ***        | ***             | ***             | ***                             |
| flattened_truck_truck_9.txt            | 1154       | trucks          | 52.116          | 149016.286                      |

Total C++ and OpenMP execution time: 149016.286 milliseconds

### Epoch Results for Cifar10

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.5192 | 44.13%   |
| 2     | 1.0281 | 63.49%   |
| 3     | 0.8128 | 71.29%   |
| 4     | 0.6679 | 76.87%   |
| 5     | 0.5501 | 80.66%   |

The sparsity of weight and bias:

| File                    | Sparsity |
|-------------------------|----------|
| features_0_weight.txt   | 0.00     |
| features_3_weight.txt   | 0.00     |
| features_6_weight.txt   | 0.00     |
| features_8_weight.txt   | 0.00     |
| features_10_weight.txt  | 0.00     |
| classifier_weight.txt   | 0.00     |


## Epoch Results for Cifar10 99% Pruned

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.5269 | 44.38%   |
| 2     | 1.0893 | 61.34%   |
| 3     | 0.9069 | 68.28%   |
| 4     | 0.7697 | 73.07%   |
| 5     | 0.6670 | 76.77%   |

The Sparsity of weight and bias:

| File                    | Sparsity |
|-------------------------|----------|
| features_0_weight.txt   | 0.42     |
| features_3_weight.txt   | 0.73     |
| features_6_weight.txt   | 0.76     |
| features_8_weight.txt   | 0.75     |
| features_10_weight.txt  | 0.71     |
| classifier_weight.txt   | 0.34     |

</small>
## Structure of Alexnet Cifar-10
```python

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjusted for 32x32 input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        relu_output = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

```
