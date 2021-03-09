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