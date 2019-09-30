# 2-6_sentiment_analysis

## Quick Start

### Download the sentiment dataset

Inside this folder execute the following commands:

```
mkdir data && cd data
wget https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/sentiment-analysis-network/labels.txt
wget https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/sentiment-analysis-network/reviews.txt
```

### Execute one of the provided scripts

Originally, the project had to be solved without PyTorch. Execute the original solution with this command that will display the iterative development of the network (see description below):

```
cd 2-6_orig
python sentiment_analysis_full.py
```

Alternatively, execute the final network (see Version 4 in the description below):

```
cd 2-6_orig
python network_v4.py
```

In the latter case you can change the two parameters at the beginning of the file.

I also provide a **PyTorch** implementation of the final network:

```
cd 2-6_pytorch
python network_v4.py
```

Here you can change the the two previous parameters and the learning rate at the beginning of the file.

## Description

The dataset in this project is small enough to be trained on the CPU.

If you run the script with the full solution you will first see some output containing information about the dataset. Then 4 different networks will be trained, illustraiting different behaviour depending on the preprocessing of the training data.

### Version 1

The input for the first version is is a vector of size 74074. 74074 is the number of unique words in the dataset. Each element of the vector is a count of how many times the corresponding word from the vocabulary appeared in the review.

The output looks something like this:

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
Progress:10.4% Speed(reviews/sec):795.9 #Correct:1247 #Trained:2501 Training Accuracy:49.8%
Progress:20.8% Speed(reviews/sec):785.1 #Correct:2380 #Trained:5001 Training Accuracy:47.5%
Progress:31.2% Speed(reviews/sec):785.8 #Correct:3638 #Trained:7501 Training Accuracy:48.5%
Progress:41.6% Speed(reviews/sec):781.3 #Correct:5066 #Trained:10001 Training Accuracy:50.6%
Progress:52.0% Speed(reviews/sec):780.2 #Correct:6388 #Trained:12501 Training Accuracy:51.0%
Progress:62.5% Speed(reviews/sec):779.0 #Correct:7638 #Trained:15001 Training Accuracy:50.9%
Progress:72.9% Speed(reviews/sec):778.4 #Correct:8888 #Trained:17501 Training Accuracy:50.7%
Progress:83.3% Speed(reviews/sec):778.4 #Correct:10138 #Trained:20001 Training Accuracy:50.6%
Progress:93.7% Speed(reviews/sec):778.6 #Correct:11388 #Trained:22501 Training Accuracy:50.6%
Progress:99.9% Speed(reviews/sec):778.5 #Correct:12137 #Trained:24000 Training Accuracy:50.5%
```

As you can see, the network is not really learning, since the accuracy is around 50% (random guessing between a review being positive or negative).

### Version 2

In the second version the instructor teaches us how to better preprocess the data. The input is still a vector of size 74074, but this time each element is 0 or 1 and indicates whether or not a word from the vocabulary appeared in the review.

The output looks something like this:

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
Progress:10.4% Speed(reviews/sec):829.9 #Correct:1949 #Trained:2501 Training Accuracy:77.9%
Progress:20.8% Speed(reviews/sec):823.2 #Correct:3994 #Trained:5001 Training Accuracy:79.8%
Progress:31.2% Speed(reviews/sec):824.3 #Correct:6121 #Trained:7501 Training Accuracy:81.6%
Progress:41.6% Speed(reviews/sec):825.4 #Correct:8278 #Trained:10001 Training Accuracy:82.7%
Progress:52.0% Speed(reviews/sec):825.8 #Correct:10439 #Trained:12501 Training Accuracy:83.5%
Progress:62.5% Speed(reviews/sec):824.3 #Correct:12580 #Trained:15001 Training Accuracy:83.8%
Progress:72.9% Speed(reviews/sec):823.8 #Correct:14694 #Trained:17501 Training Accuracy:83.9%
Progress:83.3% Speed(reviews/sec):823.0 #Correct:16868 #Trained:20001 Training Accuracy:84.3%
Progress:93.7% Speed(reviews/sec):822.6 #Correct:19055 #Trained:22501 Training Accuracy:84.6%
Progress:99.9% Speed(reviews/sec):822.2 #Correct:20375 #Trained:24000 Training Accuracy:84.8%

Test the trained network:
Progress:99.9% Speed(reviews/sec):1689. #Correct:860 #Tested:1000 Testing Accuracy:86.0%
```

Now we see that the network is learning and achieves a testing accuracy of 86%.

### Version 3

In this version the instructor focuses on speeding up the training process by eliminating unnecessary multiplications with 0 and 1.

The output looks something like this:

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
Progress:10.4% Speed(reviews/sec):2895. #Correct:1949 #Trained:2501 Training Accuracy:77.9%
Progress:20.8% Speed(reviews/sec):2793. #Correct:3994 #Trained:5001 Training Accuracy:79.8%
Progress:31.2% Speed(reviews/sec):2790. #Correct:6121 #Trained:7501 Training Accuracy:81.6%
Progress:41.6% Speed(reviews/sec):2791. #Correct:8278 #Trained:10001 Training Accuracy:82.7%
Progress:52.0% Speed(reviews/sec):2780. #Correct:10439 #Trained:12501 Training Accuracy:83.5%
Progress:62.5% Speed(reviews/sec):2769. #Correct:12580 #Trained:15001 Training Accuracy:83.8%
Progress:72.9% Speed(reviews/sec):2763. #Correct:14694 #Trained:17501 Training Accuracy:83.9%
Progress:83.3% Speed(reviews/sec):2767. #Correct:16868 #Trained:20001 Training Accuracy:84.3%
Progress:93.7% Speed(reviews/sec):2757. #Correct:19055 #Trained:22501 Training Accuracy:84.6%
Progress:99.9% Speed(reviews/sec):2755. #Correct:20375 #Trained:24000 Training Accuracy:84.8%

Test the trained network:
Progress:99.9% Speed(reviews/sec):4463. #Correct:860 #Tested:1000 Testing Accuracy:86.0%
```

Now the network is able to process around 2700 reviews per second as opposed to around 800 in the previous version, while still achieving the same classification rate.

### Version 4

In this last version the instructor teaches us how to reduce noise in the data. Our input vector is now shorter as we only consider words that appear at least a certain amount of times in the data (min\_count parameter). Further, we only use words whose log positive-negative ratio is at least a certain amount away from 0 (polarity\_cutoff parameter).

With a min\_count of 100 and a polarity\_cutoff of 0.05, the output looks something like this:

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
Progress:10.4% Speed(reviews/sec):3690. #Correct:1941 #Trained:2501 Training Accuracy:77.6%
Progress:20.8% Speed(reviews/sec):3700. #Correct:3987 #Trained:5001 Training Accuracy:79.7%
Progress:31.2% Speed(reviews/sec):3700. #Correct:6105 #Trained:7501 Training Accuracy:81.3%
Progress:41.6% Speed(reviews/sec):3685. #Correct:8251 #Trained:10001 Training Accuracy:82.5%
Progress:52.0% Speed(reviews/sec):3688. #Correct:10403 #Trained:12501 Training Accuracy:83.2%
Progress:62.5% Speed(reviews/sec):3711. #Correct:12520 #Trained:15001 Training Accuracy:83.4%
Progress:72.9% Speed(reviews/sec):3714. #Correct:14661 #Trained:17501 Training Accuracy:83.7%
Progress:83.3% Speed(reviews/sec):3720. #Correct:16841 #Trained:20001 Training Accuracy:84.2%
Progress:93.7% Speed(reviews/sec):3741. #Correct:19023 #Trained:22501 Training Accuracy:84.5%
Progress:99.9% Speed(reviews/sec):3747. #Correct:20346 #Trained:24000 Training Accuracy:84.7%

Test the trained network:
Progress:99.9% Speed(reviews/sec):5550. #Correct:847 #Tested:1000 Testing Accuracy:84.7%
```

With a min\_count of 100 and a polarity\_cutoff of 0.8, the output looks something like this:

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
Progress:10.4% Speed(reviews/sec):25603 #Correct:2022 #Trained:2501 Training Accuracy:80.8%
Progress:20.8% Speed(reviews/sec):26259 #Correct:4047 #Trained:5001 Training Accuracy:80.9%
Progress:31.2% Speed(reviews/sec):25918 #Correct:6091 #Trained:7501 Training Accuracy:81.2%
Progress:41.6% Speed(reviews/sec):25842 #Correct:8178 #Trained:10001 Training Accuracy:81.7%
Progress:52.0% Speed(reviews/sec):25720 #Correct:10272 #Trained:12501 Training Accuracy:82.1%
Progress:62.5% Speed(reviews/sec):25652 #Correct:12356 #Trained:15001 Training Accuracy:82.3%
Progress:72.9% Speed(reviews/sec):25524 #Correct:14435 #Trained:17501 Training Accuracy:82.4%
Progress:83.3% Speed(reviews/sec):25474 #Correct:16579 #Trained:20001 Training Accuracy:82.8%
Progress:93.7% Speed(reviews/sec):25437 #Correct:18709 #Trained:22501 Training Accuracy:83.1%
Progress:99.9% Speed(reviews/sec):25450 #Correct:19976 #Trained:24000 Training Accuracy:83.2%

Test the trained network:
Progress:99.9% Speed(reviews/sec):13229 #Correct:832 #Tested:1000 Testing Accuracy:83.2%
```

In both cases we sacrifice some accuracy, but gain (a lot) of training speed. According to the instructor this is what we want if we have a larger dataset: train faster. On the other hand, if the dataset was larger, the accuracy would also improve.

### Comparison of the original implementation and PyTorch

With a min\_count of 100 and a polarity\_cutoff of 0.2, the **original** output looks something like this:

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
Progress:10.4% Speed(reviews/sec):3873. #Correct:1957 #Trained:2501 Training Accuracy:78.2%
Progress:20.8% Speed(reviews/sec):3800. #Correct:4011 #Trained:5001 Training Accuracy:80.2%
Progress:31.2% Speed(reviews/sec):3786. #Correct:6124 #Trained:7501 Training Accuracy:81.6%
Progress:41.6% Speed(reviews/sec):3865. #Correct:8265 #Trained:10001 Training Accuracy:82.6%
Progress:52.0% Speed(reviews/sec):3842. #Correct:10419 #Trained:12501 Training Accuracy:83.3%
Progress:62.5% Speed(reviews/sec):3827. #Correct:12562 #Trained:15001 Training Accuracy:83.7%
Progress:72.9% Speed(reviews/sec):3858. #Correct:14687 #Trained:17501 Training Accuracy:83.9%
Progress:83.3% Speed(reviews/sec):3835. #Correct:16862 #Trained:20001 Training Accuracy:84.3%
Progress:93.7% Speed(reviews/sec):3817. #Correct:19048 #Trained:22501 Training Accuracy:84.6%
Progress:99.9% Speed(reviews/sec):3809. #Correct:20363 #Trained:24000 Training Accuracy:84.8%

Test the trained network:
Progress:99.9% Speed(reviews/sec):5705. #Correct:856 #Tested:1000 Testing Accuracy:85.6%
```

With a min\_count of 100 and a polarity\_cutoff of 0.2, the **PyTorch** output looks something like this (with a learning\_rate of 0.0001):

```
Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Trained:1 Training Accuracy:0.0%
Progress:10.4% Speed(reviews/sec):960.7 #Correct:1925 #Trained:2501 Training Accuracy:76.9%
Progress:20.8% Speed(reviews/sec):967.6 #Correct:4029 #Trained:5001 Training Accuracy:80.5%
Progress:31.2% Speed(reviews/sec):976.6 #Correct:6190 #Trained:7501 Training Accuracy:82.5%
Progress:41.6% Speed(reviews/sec):981.4 #Correct:8358 #Trained:10001 Training Accuracy:83.5%
Progress:52.0% Speed(reviews/sec):975.6 #Correct:10538 #Trained:12501 Training Accuracy:84.2%
Progress:62.5% Speed(reviews/sec):1027. #Correct:12713 #Trained:15001 Training Accuracy:84.7%
Progress:72.9% Speed(reviews/sec):1072. #Correct:14888 #Trained:17501 Training Accuracy:85.0%
Progress:83.3% Speed(reviews/sec):1106. #Correct:17106 #Trained:20001 Training Accuracy:85.5%
Progress:93.7% Speed(reviews/sec):1136. #Correct:19320 #Trained:22501 Training Accuracy:85.8%
Progress:99.9% Speed(reviews/sec):1148. #Correct:20651 #Trained:24000 Training Accuracy:86.0%

Test the trained network:
Progress:99.9% Speed(reviews/sec):4806. #Correct:865 #Tested:1000 Testing Accuracy:86.5%
```

Notice the slightly better performance of the PyTorch version. However, I did not try to tweak the learning rate of the original implementation, so it might also perform better. Unfortunately, the PyTorch version is much slower. This is due to the manual optimization of computation steps in the original implementation. I did not try to optimize the PyTorch version at all.

