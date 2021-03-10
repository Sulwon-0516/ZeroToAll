# Report

- Date : 2.Mar.21 ~ 9.Mar.21
- Content : Implementation practice with Pytorch ZeroToAll

    → Chapter 8, 10, 11, 12

- **Chapter 8 DataLodaer**
    - Data Loader practice
    - kaggle data set loading + preprocessing of .csv file.
    - Using Torchvision dataloader

- **Chapter 10 Basic CNN**
    - Implement simple LeNet.
    - Using LR_scheduler
    - Saving + Loading checkpoints

- **Chapter 11 Advanced CNN**
    - Implement Inception v4
        - Used tiny ImageNet (64x64)
        - Upscaled into (226x226) and checked the loss reduces.
    - DenseNet
        - check detail in [DenseNet.md](http://densenet.md)
        - achieced 99+ % in 50*200 train images. (checked it overfits well)

- **Chapter 12 RNN**
    - As I'm not friendly with RNN, I've done the exercises in ppt.
    - Simply training RNN with embedding vector + output fully-connected layers
    - Spent lots of time for data preprocessing...

- **Chapter 13 RNN II**
    - Sequence input → classification output.
    - Used given dataset (name <> country)
    - Used LSTM with 100 hidden-layers with depth 2.
        - The initial loss ~= 1.8 but it doesn't decrease
        - I changed, batch, LR and etc. but nothing changed..
        - Initial accuracy is about 48 %

    - Dataset Characteristics

        {'Polish': 0, 'Japanese': 1, 'Arabic': 2, 'Scottish': 3, 'Russian': 4, 'Irish': 5, 'German': 6, 'Czech': 7, 'Portuguese': 8, 'Spanish': 9, 'Korean': 10, 'Chinese': 11, 'Greek': 12, 'Dutch': 13, 'Vietnamese': 14, 'English': 15, 'Italian': 16, 'French': 17}
        [  92.  660. 1333.   66. 6272.  154.  482.  346.   49.  198.   62.  178.
        135.  198.   48. 2445.  472.  184.]

        Half of the names are Russian.... 

        So, it just predict the result it Russian, and getting 48% loss.

        However, it fails to avoid this local minima (maybe?)

    - Why it doesn't work?
        - Maybe my code has some problem.
            - Yet, I can't understand how LSTM & RNN module in torch works.
                - They get (Sequence x Batch x Input shapes) as input...
                - I thought something like this

                ```jsx
                hidden = model.init_hidden()
                for i in Input:
                	out, hidden = model(i,hidden)	
                ```

                - However, lots of code just write like this.

                ```jsx
                hidden = model.init_hidden()
                out, hidden = model(Input,hidden)
                ```

                - Also, lots of codes use "hidden as result", not "output as result" for softmax classifier.

            - RNN, LSTM modules are so tricky..
        
        - Small Dataset Overfitting
            - Tested with 198*2 datasets.
            - Final accuracy was around 90%...

                → No LR-Decay... (but Adam...)

                → embedding to 20 dimension → 100 Hidden depths, 200 epoch results.

                → Maybe, more training time required?

            → Found out problem on collate_fn.

            - Code error : the maximum length is fixed as 29. (average name length ~ 8, so 2/3 are filled with <pad>)

        - Large dataset training.
            - As i fixed it, tried again, but failed. (still stuck on 0.48)
            - With Adam, tried with diverse LR and LR_scheduler → but failed.

            - Changed to SGD with 0.9 momentum but doesn't work.
                - it tends to not depend on the INPUT data...

            - Changed to RMSProp with alpha = 0.9
                - With LR = 2e-4 achieved about 0.7 accuracy.
                - However decaying stopped around that value, so added LR_decay with factor 0.2 at epoch 40.
                - But, after LR decay, the loss increased from 0.9 → 1.3 (acc 0.7 → 0.5)

                    And no anymore decaying.

            - Finally, I tested without LR decay, so around epoch 70, I get 76% acc

                →retrain_156_0.0002_30-1.pt

                - However, the loss increased after epoch 80, and stopped decaying.
                (loss 1.0 → 1.2)

                - about result...

                    I used Depth 3 LSTM (20,200) and two linear layers before input and after output (they are (29,20) and (200,18))

                    so the total # of parameters = 3(layers)*4(4000+40000+200) + 580 + 3600 = 176800 + 4180 ~ 500k paramsI think the model is big enough to learn...

                    there must some reason that it fails. Especially when LR decay is applied, the loss increases and never return back.

                    ( I decayed LR with 0.2 so in my opinion, the loss should reduced to the previous value in 10 epochs, but it doesn't recover.


