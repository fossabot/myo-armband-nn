# myo-armband-nn
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkyr7%2Fmyo-armband-nn.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkyr7%2Fmyo-armband-nn?ref=badge_shield)

Gesture recognition using [myo armband](https://www.myo.com) via neural network (tensorflow library).
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/myo-armband-nn-logo.jpg)


## Requirement
**Library** | **Version**
--- | ---
**Python** | **^3.5**
**Tensorflow** | **^1.1.0** 
**Numpy** | **^1.12.0**
**sklearn** |  **^0.18.1**
**[myo-python](https://github.com/NiklasRosenstein/myo-python)** |  **^0.2.2**


## Collecting data
You can use your own scripts for collecting EMG data from Myo armband.
But you need to push 64-value array with data from each sensor.<br />
By default myo-python returns 8-value array from each sensors.
Each output return by 2-value array: ```[datetime, [EMG DATA]]```.<br />
64 - value array its 8 output from armband. Just put it to one dimension array.
So you just need to collect 8 values with gesture from armband (if you read data 10 times/s its not a problem).

In repo are collected dataset from Myo armband collected by me. Dataset contains only 5 gestures:
```
👍 - Ok    (1)
✊️ - Fist  (2)
✌️ - Like  (3)
🤘 - Rock  (4)
🖖 - Spock (5)
```

## Training network
```sh
python3 train.py
```
75k iteration will take about 20 min on GTX 960 or 2h on i3-6100.

## Prediction
### Prediction on data from MYO armband
```sh
python3 predict.py
```
You must have installed MYO SDK.
Script will return number (0-5) witch represent gesture (0 - relaxed arm).

### Prediction on test dataset
```sh
python3 predict_test_dataset.py
```
Example output:
```
Accuracy on Test-Set: 90.74% (49 / 54)
[14  0  0  0] (0) Fist
[0 9 0 2] (1) Paper
[ 0  0 13  1] (2) Scissors
[ 0  0  2 13] (3) Ok

```

## Model
| **Fully connected 1 (528 neurons)** |
| :---: |
| ReLu |
| **Fully connected 2 (786 neurons)** |
| ReLu |
| **Fully connected 3 (1248 neurons)**  |
| ReLu |
| Dropout |
| **Softmax_linear** |
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/myo-armband-nn-model.png)

## License
[GNU General Public License v3.0](https://github.com/exelban/myo-armband-nn/blob/master/LICENSE)


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fkyr7%2Fmyo-armband-nn.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fkyr7%2Fmyo-armband-nn?ref=badge_large)