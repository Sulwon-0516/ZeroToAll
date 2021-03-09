# DenseNet

- Dense Block
    - In the dense block, ALL layers have direct connection to subsequent layers.
        - Concat in Depthwise
        - Unit : BN → ReLU → (1x1 Conv (4k output) → BN → ReLU→) 3x3 Conv (size preserve)

            → 1x1 bottleneck : feature reduction 

            → similarly, can apply compression rate.

        - Each unit : narrow Depths ( like k=12 (k : growth rate ) 
        → ok because of concat

- Between Dense Blocks → Pooling
    - For Downsampling
    - BN → 1x1 Conv → 2x2 Avg Pooling (stride = 2)

- 3 Dense Block
    - 32x32, 16x16, 8x8 Feature maps in blocks.
    - Start with 3x3 zero-padded depth : 16
    - End with Global Avg Pooling + softmax classifier

    → when CIFAR

- 4 Dense Block
    - resize to 224x224
    - 7x7 Conv with stride 2 + 3x3 Max Pooling with stride 2
    - 56x56, 28x28, 14x14, 7x7
    - Depth : 6→12→24→16

    → when ImageNet

I want to make a model generator which can freely control the L and k.

<real Implementation issue>

64 → 64 →32 →16 → 8 → 4 : Avg Pooling

→ to make it, set the first stride as 1 not 2.

<Training Details>

- SGD(momentum : 0.9, weight_decay : 0.0001)
- Batch size : 64
- epochs : CIFAR : 300
- LR : 0.1, 0.01 at half, 0.001 at 3/4

If it doesn't apply data augmentation, add drop out after all Conv with 0.2

In my Opinion, to get total equivalency, change the last FCL softmax to the 

A . 1x1 depth : # of class

B. Average Pooling on the whole feature map. 

C. softmax.

→ When the image size is big, it can make some issues..... 

→ So It's essential to divide the whole model in two scales.

Q1. What is the depth of the Transition layer?

→ It's the compression rate.

Q3. Why only horizontal flip. why no vertical flip?

→ depends on the datatype.

<Result>

Tested with following parameters.

Tested on subset of tiny imageNet

(50images for each class for training)

Finally, 

train accuracy was almost 99%.

However, # of training set : 50, # of validation set : 50 so it's hard to get good result without any regularization methods.

Validation result :

acc : tensor(0.0048)

→ almost similar as random selection... 

→ It's because I didn't applied data augmentation without flipping. 

→ Paper recommend to add Dropout if not using the data augmentation... but I didn't applied...