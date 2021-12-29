# Scene Recognition

## Pretrained model
Based on Resnet50.

## Labels
Sitting, kitchen, corridor (see Place365 dataset)

## Requirements
Please refer to [requirements.txt](https://github.com/MorphSeur/Action-PostureRecognition/blob/master/requirements.txt).

## Notebook
### mainSceneRecognition.ipynb
- First cell splits the input video located in ./input/ to frames stored ./input/frame/.
- Second cell read frame by frame located in ./input/frame/, predict labels, put predicted labels in the dedicated frame as text.
- Third cell read the recognized frames located in ./input/frame/ and build video stored in ./output.

## Issues
Lack to recognize the kitchen location in some videos.