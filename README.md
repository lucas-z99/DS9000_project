# DS9000 Project


## Download Dataset

HWDB1.1 from [Hugging Face](https://huggingface.co/datasets/OrkaZeta/HWDB1/tree/main)
- [TrainPart1.zip]( https://huggingface.co/datasets/OrkaZeta/HWDB1/resolve/main/Gnt1.1TrainPart1.zip)
- [TrainPart2.zip]( https://huggingface.co/datasets/OrkaZeta/HWDB1/resolve/main/Gnt1.1TrainPart2.zip)
- [Test.zip]( https://huggingface.co/datasets/OrkaZeta/HWDB1/resolve/main/Gnt1.1Test.zip)

Unzip here:

```text
DS9000_project/
└── raw_dataset/
    ├── HWDB1.1trn_gnt/
    └── HWDB1.1tst_gnt/
```

## About this dataset

Training set:

 - 938679 Examples, by 240 Writers

 - 3922 Unique charactors

 - Image dim (max = ~256x256)

<img src="./others/image-dim-plot.png" alt="Description" width="500" height="450">

## Prepare Data

Run `a1_prepare_data.py`

 - Prepared data is packed into shards under `prepared/<size>/train`, `/val`, `/test`

 - `val_set` is randomly picked by Writer (not by examples). Since if the model is to generalize, it needs to read handwriting from someone new.


## Eyeball the Data

Run `a2_check_shard.py` to unpack one shard in `/debug`:
 - eg set ```writer = "1001-f"``` to see first guy's hand writing
 - eg set ```writer = "1052-f"```







## Models

### The Classifier
Standard Fully connected network. See `NN_Clasifier.py`

 - Test run #1: loss=3.5, accuracy=24.3% (ReLU / 2500 loops / n_x = 128x128 / n_y = 3900). Stopped due to taking too long! Will resize image and try a longer training session.

### The CNN
Standard CNN. All hail the matrix machine god.

(WIP)

### The Verifier
Is Siamese NN a good fit?

(WIP)


## TensorBoard

To use the Dashboard, run in console:

    tensorboard --logdir runs

Then open in browser:

    http://localhost:6006/    

<img src="./others/tensorboard.png" alt="Description">


## STOP training midway

To stop a training session, create a file named "STOP" in the model folder (eg `\runs\24_1000-1200_20261115_074744`), the code will exit after finishing the current epoch.

Just ctrl+c to exit is usually safe too (If you are very+very+very unlucky this can currput log/model file)


## Image Size Study


 - Big image size = overfitting early
 - This is an unintentional effect of larger input. A red model (128) has 4x more weight in layer_1 than purple model (64), but the charactors only provides about the same patterns to learn, so the model has a lot more room to memorize.
 - This implies there is a race between "memorization"(M) and "generalization"(G).
 - A model is always simultaneously memorizing and generalizing. And if M peaks earlier, then the result plateau is lower (eg. all purple plateau is higher than any red). 
 - So we should deny M, to allow G to reach full saturation, then stop training immediately. 
 - This explains why various techniques all sum-up to "add noise / randomness / chaos" to the sauce. Chaos is asymmetric. You can't memorize chaos (by def), but you might learn how to mitigate it, or even in this case, utilize it.


<img src="./others/size-study-y200.png" alt="Description">



## Random Idea
 - Normalize color? if some dark, some brighter
 - Rid of RBG? Since it's basically grayscale
 - Right now val_set is correctly picked by writer. But maybe we can intentionally mistake to pick by samples, so to see if it really mess up (leak) the model?
