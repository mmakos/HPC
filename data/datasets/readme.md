# Final datasets

## Static poses
Accuracy on this datasets is 91.84%. Trained on batch size = 1000. 

### Training
* Name - Static.npz
* Poses: stand, sit, lie, lean, kneel
* Samples - 6000 samples per label
* Augmentation - 800 XY rotations of 10-15 static images

### Validation
* Name - StaticVal.npz
* Poses: stand, sit, lie, lean, kneel
* Samples - 1000 samples per label
* Filled keypoints

## Dynamic poses
Accuracy on this datasets is 86.70%. Trained on batch size = 64.

### Training
* Name - Dynamic.npz
* Poses: walk, jump
* Samples - 6000 samples per label
* Augmentation - XY rotations

### Validation
* Name - DynamicVal.npz
* Poses: walk, jump
* Samples - 1000 samples per label
* Filled keypoints

## All poses
This datasets are basically combined static and dynamic datasets.
Accuracy on this datasets, when using one model is only 69.60%. Use hybrid solution using static and dynamic model together. 

### Training
* Name - All.npz
* Poses: stand, sit, lie, lean, kneel, walk, jump
* Samples - 6000 samples per label
* Augmentation - XY rotations

### Validation
* Name - AllVal.npz
* Poses: stand, sit, lie, lean, kneel, walk, jump
* Samples - 1000 samples per label
* Filled keypoints
