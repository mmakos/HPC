# Human Pose Classification - BEng Thesis
![WEiTI - elka.pw.edu.pl](https://lh3.googleusercontent.com/fife/ABSRlIqJJC3S6Kcy0WmhYjwolt76L3_JZfK_Mn_Yb9h97Uif2ZgtiZZgPDDxUv_45mHQCC49TI-lL4ml8IJR1dY6Xtajf2w0tqbeXdO_tmvoS1luu09V93tY5ayOjWtPc5Cg7UY6MkaMkJro2g0QLELHvnXolwDp1xGHiNKoN4r9d3vwWzwwJX1ftmiN3Q6OqkX046iC0S7tyEVHLc0untMLxFNd6Q__gmsP3FueFFcDGDt-vYuNhNB9knOh4OSQMQOk2xhGQ5_FHUvCtj4r6PocZitQ2qWADQTts8CoCACNwCq2x7PaKSB9Qmp981kG_yjfDafJGlxIEwT3Ktkhov7XuHW8rZBjgxjUFU2eeU7GjyAsocD2m9HSoEkeF_EU_MdyoaHdlaHs8GKxu9XzSlqYl47_LgQxRNGOx14MOwLiFIVD3FsOR4n9FNwrX3tMJ0OQCkto2heHdUFZso8LpmrdSuuRfoWnS0c3STfx8w6T9wgi5hcALVjfhZ8xU4EUWUS-wig-BrZLClS_II01KlyhimbALnNJvp1PncFmB9aAYD5FJD9hKH9TM67a2kRPceX72pPfw1WCd6YzcP-6qccMHZfgijo9R5w0VGHEh0IRP2SPDFIkWwSsIpDyCGyBxK02y_7nDirbohezNs26EYj8O5EQ42Ofkr2kF9X9KOEa8xltuBJq6lIZGhjuxCoXj5NroTBA-kvFyLN28pPBT7bfQa2ayQt_l6fPqQ=w2560-h699-ft)

## About project
Created system has to recognize human static and dynamic poses such as standing, walking, running itd.
As an input it takes video stream of RGBD camera (color and depth stream) from a robot.
Then, based on classified poses we can monitor human movement and e.g. detect falls.

## Architecture basics
Solution is based on body keypoints which are detected using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) library.
System estimates human body's keypoints with OpenPose and converts them into small abstract image which represents theirs positions in 3D space trough multiple frames (so it's in facet 4D space).
Columns of this image represents sequent frames, rows represents different keypoints and color represent 3D position (B-x, G-y, R-D).
So example image for 15 keypoints and only RGB camera looks like:<br>
<img src="https://lh3.googleusercontent.com/fife/ABSRlIpLfqeo1-oiIlSGdv8EDmmfh5r3KCjr4W0qMHZIiHAreUn6tMpD96VcIdcqVU9amQwbAn4iDwQvvMG8mVGdKMfTx3tF07r_IjQf_oADpRYe4_tpGfiAOgffmZPLXOP_DQgj30gr4ByUtyEUTv1sbJXLF4xCBNWCXpPVx2xghJEe7t8_I9YNaSuRcbx9oWL8bot5GG3zDYWBXrJA0QougJn3X9HiA6RB3vIrUWdqbRZ02qnHIdcOopCi9_4Yyq_LXzaLTMtyRgkcGqzUt2o_FlzgxPRuo-0UbjUBo5mc28aYOSUckhDA_cguR6Y28VpKSPwXym4gwX9xTX3UsSVJfEPCQIehelRS5Reh6p_sbtCrC__7Pqa_potvjc7YY0u4ofoLgRvugfQC4Oko1Sf9iXskpB_K7yPy2bu5CjERIHPf9Kp2KLYTkPf7t84igcF5LEr7ssXijWUVPmawKkqQ6K8PwxOx44qt_pqipKR4xyzT97qsaVRG9DLVFW4A5zENzjs81el3JpjW2TOfjIxAvzAsxFWSdvbk5ADMzWvZUO_vqNLhxdsd-HzO4ZtTQo8WcANZl9AdDxhZyBaWwywu-I5q27E6EFucIMhYHSnZ6vBvoth54YxAIDeyyW6Pk1BRDXP0iAKH7cKjZFzmqr8zlSKK8N6POOMn6o7zwmGgB_PhMfloNu9rFWDCOjKro-I2rY9HrUtulEGPqRhaGhv33AL0pcNIz-dFxA=w1278-h949-ft" alt="Example coded keypoints image" width="320"/>
<br>Black fields represents not detected keypoints.

## Requirements
For now:
* Built OpenPose in folder `/externals/openpose/build`
* Python 3 (I work with 3.7 and I don't support other versions for now (e.g. OpenPose didn't work for me with Python 3.9) ) with libraries:
    * tensorflow
    * opencv-python
    * primesense
    * keyboard
    * numpy (comes with opencv)<br>
      *You can satisfy all above requirements by running script `requirements.bat`*
* For working with robot camera - Orbbec Astra SDK - OpenNI in folder `/externals`

## Working with (available modules)
### Data processing
#### recordVideo.py
Module records stream of RGBD camera and writes output to *.oni* file. Output video will be stored in `/data/videos`.

Usage: `python recordVideo.py -v video_name`:
* *video_name* - name of output video without extinction.

#### proceedVideo.py
Module takes recorded video in *.oni* or regular video format, estimates human skeletons and converts this skeletons into images.
Module also shows video with estimated skeletons.

Usage: `python proceedVideo.py video_path -d -v -p proceed`:
* video_path - path to your video relative to running folder or to `/data/video` folder.
* -d - if this option is selected, module will proceed depth frames as well. Select this option only when you proceed *.oni* file.
* -v - view mode. Select this option when you only want to view your video (without estimating skeletons. Useful for viewing *.oni* files.
* proceed - proceed mode. Select this option when you want to code estimated skeletons to images. *Proceed* is name of folder where images will be saved (relative to `/data/images/` folder).

#### viewImagesAsVideo.py
Module shows skeleton images as video.

Usage: `python viewImagesAsVideo.py -s skeleton -f fps -z zoom`
* skeleton - index of skeleton to be shown
* fps - how many frames per second will be played
* zoom - factor by which every dimension will be multiplied

#### createDataset.py
Module create dataset from images to numpy file ready to train.

To properly generate dataset you have to store your images in folders named `<filename>_<label>` where *filename* can be anything but *label* is label of images according to *poses* table in *consts.py* file.
Then you have to move all this folders to move all this folders to one final folder. So it should look like this:
<pre>
poses
|--run_2
|  |--image1.png
|  |--image2.png
|--stand_0
|  |--image1.png
|--walk_1
</pre>

Usage: `python createDataset.py dataset_name -p poses -z`
* dataset_name - name of dataset you want to create.
* poses - path to your folder with labels subfolders relative to `/data` folder.
* -z - if this option is selected dataset will be stored in *.zip* format instead of *.npy*.

### Training
#### train.py
Module opens model or creates new one, trains it on given dataset in *.npz* format and saves trained model.

Usage: `python train.py dataset_name -m model_name -o output_model`
* dataset_name - name of data set on which you want to train your model.
* model_name - name of model, which will be opened or created if no such model exists
* output_model - name of output model, if model with that name exists it will be overwritten. If this arg is not specified model won't be saved.

### Running
#### estimateVideo.py
Module loads video and estimates poses for every human in every frame.

Usage: `python proceedVideo.py video_path -m model -d -v -p`:
* video_path - path to your video relative to running folder or to `/data/video` folder.
* model - name of model you want to load relative to `/data/models` folder.
* -d - if this option is selected, module will proceed depth frames as well. Select this option only when you proceed *.oni* file.
* -v - view mode. Select this option when you only want to view your video (without estimating skeletons. Useful for viewing *.oni* files.
* -p - proceed mode. Select this option when you want estimate poses.

### Internal modules
#### const.py
Module stores all constants:
* *frameWidth, frameHeight* - color frame dimensions, automatically set after running video stream
* *depthWidth, depthHeight* - depth frame dimensions, automatically set after running video stream with depth canal
* *frameDepth* - maximal depth of depth frame
* *framesNumber* - number of frames stored in one skeleton image
* *keypoints* - number of keypoints stored in one skeleton image
* *keypointThreshold* - threshold from which keypoints are proceeded
* *poses* - table of poses (labels)

#### frame.py
Class Frame stores all skeletons.
* *Frame.proceedFrame( humans )* - function proceeds single frame by updating tracked skeletons and estimate pose for every tracked skeleton. It returns array, where first element is array of pose probabilities and second element is id of skeleton.
* *Frame.proceedHuman( human, newSkeletons )* - internal function proceed detected skeletons
* *Frame.classifyPose( skeleton )* - function returns probability map for given skeleton
* *Frame.getSkeletons( humans )* - function proceed single frame and returns array of skeleton images. Function is equivalent od *proceedFrame* but for dataset creating.
* *getBoundingBox( keypoints )* - help function, returns size of bounding box for given keypoints in format ( width, height, depth )

#### skeleton.py
Class Skeleton stores single skeleton tracked through multiple frames.
* *Skeleton.updateSkeleton( keypoints )* - function updates skeleton with given keypoints
* *Skeleton.updateImg()* - internal function updates skeleton image by removing oldest frame and adding new one
* *Skeleton.compareSkeleton( keypoints, minDelta )* - function returns probability that given keypoints belongs to this skeleton.
* *Skeleton.getSkeletonImg()* - function returns skeleton image in numpy array format
* *Skeleton.getSkeletonId()* - function returns skeleton id. Skeletons stored in *Frame* class are not sorted, same as skeletons returned from *proceedFrame()* function.

#### model.py
Module creates network models.
* *getModel()* - function returns default model.

#### rgbdMap.py
* *mapToRGBD( keypoints, depthCanal )* - unction maps given keypoints of all humans to RGBD image. Keypoints are [ humans[ x, y, score ] ].
