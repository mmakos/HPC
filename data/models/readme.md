# Available models

## Static
* Poses: stand, sit, lie, lean, kneel
* Accuracy = 91.84%
* Loss = 0.3912
* Epoch = 13
* Batch size = 1000
* Learning rate = 0.0001
* Train dataset - Static.npz (6000 images per label)
* Validation dataset - StaticVal.npz (1000 images per label, filled keypoints)

## Dynamic
* Poses: walk, jump
* Accuracy = 86.70%
* Loss = 2.4816
* Epoch = 40
* Batch size = 64
* Learning rate = 0.0001
* Train dataset - Dynamic.npz (6000 images per label)
* Validation dataset - DynamicVal.npz (1000 images per label, filled keypoints)

## All
In order to get better accuracy, use hybrid solution using static and dynamic models together.

* Poses: stand, sit, lie, lean, kneel, walk, jump
* Accuracy = 69.60%
* Loss = 1.9346
* Epoch = 40
* Batch size = 1000
* Learning rate = 0.0001
* Train dataset - All.npz (6000 images per label)
* Validation dataset - AllVal.npz (1000 images per label, filled keypoints)