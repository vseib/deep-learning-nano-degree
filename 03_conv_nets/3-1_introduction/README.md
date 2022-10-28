# 3-1 Evaluation of created models on CIFAR



## Description

You can choose one of the three models on lines 329 to 332.
Default is the most "complex" model. The models are described as follows:
* Net1: bigger model (3xConv, 3xLinear)
* Net2: smaller model (3xConv, 3xLinear - less filters/neurons)
* Net3: final version after seeing the solution example (4xConv, 3xLinear)

Further, in line 64 you can choose to use or not to use image augmentation.
Default is with augmentation.

## Evaluation

Here you can see the different Optimizers used and the results obtained.

|Model|Optimizer|Learning Rate|Additional Info|Epochs Trained|Test Accuracy|
|---|---|---|---|---|---|
|Net1|Adam| 0.001 | dropout = 0.5 | 30 (?) | 70 %|
|Net1|SGD | 0.01  | dropout = 0.5 | 30 (?) | 72 %|
|Net1|SGD | 0.025 | dropout = 0.5 | 30 (?) | 71 %|
|Net1|Adam| 0.001 | dropout = 0.25 | 30 (?) | 71 %|
|Net1|SGD | 0.01  | dropout = 0.25 | 30 (?) | 71 %|
|Net1|SGD | 0.025 | dropout = 0.25 | 30 (?) | 71 %|

|Model|Optimizer|Learning Rate|Additional Info|Epochs Trained|Test Accuracy|
|---|---|---|---|---|---|
|Net2|Adam| 0.001 | dropout = 0.5 | 30 (?) | 68 %|
|Net2|SGD | 0.01  | dropout = 0.5 | 30 (?) | 68 %|
|Net2|Adam| 0.001 | dropout = 0.25 | 30 (?) | 67 %|
|Net2|SGD | 0.01  | dropout = 0.25 | 30 (?) | 68 %|

|Model|Optimizer|Learning Rate|Additional Info|Epochs Trained|Test Accuracy|
|---|---|---|---|---|---|
|Net3|Adam| 0.001 | dropout = 0.5 | 30 (?) | 70 %|
|Net3|SGD | 0.01  | dropout = 0.5 | 30 (?) | 74 %|
|Net3|Adam| 0.001 | dropout = 0.25 | 30 (?) | 71 %|
|Net3|SGD | 0.01  | dropout = 0.25 | 30 (?) | 74 %|
|Net3|SGD | 0.01  | dropout = 0.25, with data augmentation | 35 | 77 %|
|Net3|SGD | 0.01  | dropout = 0.25, with data augmentation | 60 | 79 %|
|Net3|SGD | 0.01  | dropout = 0.25, with data augmentation, with dilated conv | 60 | 75 %|
