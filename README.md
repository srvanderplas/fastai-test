# fastai-test

## FasterRCNN.py
### Classes and Functions
  * data_split(dataset_path, train_validate_percent = 0.9, train_percent = 0.8, test_percent = 0.1)
  * class VocDataset(Dataset)
  * Averager
  * plot_dl(batch, index=0)
  * plot_prediction(batch, predict_outputs=False, index=0, threshold=0.2)
  
## Result
### Nov 17, 2022
#### Model1
  * All-model1.pt
  * Trained with around 400 batches, batch_size = 4, 2 epochs
  * No transform
  * Predict 100 proposal regions with labels and scores, ** some of the boxes are off **
### Nov 20, 2022
#### Model2
  * All_model2.pt
  * Trained with around 400 batches, batch_size = 4, 2 epochs
  * No transform
  * Much more reasonable proposing regions
  
### Todo
  - [X] Train with a all train images
  - [X] Create a Shiny app
      - [X] Need all predicted data
      - [X] App.py can't connect to server, probably because running on server
          + solved by running on local
  - [ ] valid_dl with batch_size=4,
      * IndexError: boolean index did not match indexed array along dimension 0; dimension is 45 but corresponding boolean dimension is 36
      * Currently bypass this problem by seeting batch_size=1
  - [ ] Write a `to_json` function to save predicted outputs as Json file
      * TypeError: Object of type Tensor is not JSON serializable - dict to json error in Pytorch
      * Currently save valid_dl and predicted outputs with `torch.save(object, pt)`, can be load but not human readable.
  - [ ] Commit the model or model parameters to github
      
