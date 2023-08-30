# fastai-test

## fasterrcnn_project

### VocDataset.py

-   Latest **Done** and **Todos** please check the [Logs](https://hiiiua.github.io/#category=Logs)

<!-- -->

-   To use \`VocDataset.py\`, following directory structure has to be met.

    ``` bash
    .
    ├── ./JPEG_Images
    │   └── ./JPEG_Images/demo_image1.jpg
    ├── ./Modified_Annotations
    │   └── ./Modified_Annotations/demo_annotation1.xml
    ├── ./my_classes.json
    ├── ./test.txt
    ├── ./train.txt
    └── ./validate.txt
    ```

## app.py

-   An interactive visualization
-   Outputs from model2
-   Two images
    -   The top one is the image with predicted boxes
    -   The bottom one is the image with original boxes
-   `Number of Boxes` allows manually choosing how many predicted boxes in the predict image
-   `Number between 0 and 896` specifies the index of the image to display from all 897 images we have
-   Three tabs
    -   `Confusion Matrix` shows the heatmap of confusion matrix
    -   `Table` lists all the 13 classes we defined and the number of classes in the each of the plots. The last row `Total` is the number of each labels summed up for both origin and prediction. The predict `Total` is the `number of boxes` specified above.
    -   `Score` lists all the predict scores, the number is also specified by `number of boxes`. ![Interactive visualization](https://github.com/srvanderplas/fastai-test/blob/main/README_img/Shiny.png)

## Result

### Nov 17, 2022

#### Model1

-   All-model1.pt
-   Trained with around 400 batches, batch_size = 4, 2 epochs
-   No transform
-   Predict 100 proposal regions with labels and scores, \*\* some of the boxes are off \*\* ![100 porposing regions from model1](https://github.com/srvanderplas/fastai-test/blob/main/README_img/mode1_100_boxes.png)

### Nov 20, 2022

#### Model2

-   All_model2.pt
-   Trained with around 400 batches, batch_size = 4, 2 epochs
-   No transform
-   Much more reasonable number of proposing regions
    -   Predict boxes from model2 ![Predict boxes from model2](https://github.com/srvanderplas/fastai-test/blob/main/README_img/model2_001_pred.png)

    -   Origin boxes  
        ![Origin boxes](https://github.com/srvanderplas/fastai-test/blob/main/README_img/model2_001_origin.png)  
-   Some of the proposing regions are reasonable if we limit the number of boxed displayed
    -   Predict boxes from model2 ![Predict boxes from model2](https://github.com/srvanderplas/fastai-test/blob/main/README_img/model2_002_pred.png)

    -   Origin boxes

        ![Origin boxes](https://github.com/srvanderplas/fastai-test/blob/main/README_img/model2_002_origin.png)  
-   Some he predictions match with the original label, but Obviously, the original labels are not matching with our scheme, **need to look into input data**.

### Todo

-   [x] Train with a all train images
-   [x] Create a Shiny app
    -   [x] Need all predicted data
    -   [x] App.py can't connect to server, probably because running on server
        -   solved by running on local
    -   [x] Add confusion matrix
-   [ ] valid_dl with batch_size=4,
    -   IndexError: boolean index did not match indexed array along dimension 0; dimension is 45 but corresponding boolean dimension is 36
    -   Currently bypass this problem by seeting batch_size=1
-   [ ] Write a `to_json` function to save predicted outputs as Json file
    -   TypeError: Object of type Tensor is not JSON serializable - dict to json error in Pytorch
    -   Currently save valid_dl and predicted outputs with `torch.save(object, pt)`, can be load but not human readable.
-   [ ] Commit the model or model parameters to github
