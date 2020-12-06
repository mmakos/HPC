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
So example image for 15 keypoints, 32 frame, static pose with RGBD camera looks like:<br>
<img src="samples/f44s0.png" alt="Example coded keypoints image" width="320"/>
<br>Black fields represents not detected keypoints.

## Requirements
For now:
* Built OpenPose in folder `/externals/openpose/build`
* Python 3 (I work with 3.7 and I don't support other versions for now (e.g. OpenPose didn't work for me with Python 3.9) ) with libraries:
    * tensorflow
    * opencv-python
    * primesense (for *.oni* files and some RGBD sensors with OpenNI 2 support)
    * pyrealsense2 (for *.bag* files and RealSense sensor)
    * keyboard
    * numpy (comes with opencv)<br>
      *You can satisfy all above requirements by running script `requirements.bat`*
* For working with robot camera - Orbbec Astra SDK - OpenNI in folder `/externals`
* For working with RealSense camera - Realsense SDK.

## Working with (available modules)
For now I implemented modules to create datasets from saved RGBD video, training net and pose classification.

Dataset creating is done by estimation skeletons with OpenPose and then converting it to coded skeleton images with my algorithms (*proceedVideo* module).
Then a set of created images can be edited by hand by filtering wrong images, splitting it to labels and augmenting. Then final dataset is created to *.npz* file (compressed numpy arrays) ready to train (*createDataset* module).
Net training is done with *train.py* module.
Then you can classify pose with *estimateVideo.py*.

Both *estimateVideo.py* and *proceedVideo.py* modules works with *.bag* file (any RGBD stream saved by rospy library (used for RealSense sensor, by it can be used also for Tiago robot sensor), *.oni* files (not tested yet, doubt it will be used), regular video file (no depth frames though) and finally with images folder (images has to have on 4th position from end letter d for depth and c for color frame and has to be in alphabetical order).
*estimateVideo.py* module works with RealSense live stream as well.

Usage of above modules is described below. 
### Running
#### estimateVideo.py
Module loads video and estimates poses for every human in every frame.

Usage: `python proceedVideo.py -v video -m model -w write_name -P -p -g`:
* *video* - path to your video relative to running folder or to `/data/videos` folder. If none, program will try to run camera stream.
* *model* - name of model you want to load relative to `/data/models` folder.
* *write_name* - name of output video (if you want to save proceeded video).
* *-P* - preview mode - select this option when you only want to view your video (without estimating skeletons. Useful for viewing *.oni* files.
* *-p* - proceed mode - select this option when you want estimate poses.
* *-g* - gpu mode - tensorflow will work on GPU. This is not default setting, because OpenPose use a lot of GPU memory, so it cannot run together with tensorflow.

### Data processing
#### Dataset creation pipeline
Current general pipeline of creating dataset (pose is recorded pose and X is number of recording):
1. Create folders in /data/images and /data/videos for your data (`mkdir /data/images/example`, `mkdir /data/videos/example`). 
2. Record video as image sequence (finally it is the most universal format and the only one supported in all modules) with only one main skeleton:
    * `python recordVideo.py -v example/poseX -c` (-c for color preview).
    * press `s` to start recording (when camera is focused and you are ready)
    * press `q` to end recording
    * remember that path to your video must have slash on the end (`example/poseX/`)
3. Estimate keypoints and get annotation file:
    * `python proceedVideo.py example/poseX/ -p example/pose -k`
    * rename main skeleton file from s_atY.p to poseXatY.png and delete rest of skeletons (or you can rename it to poseX+1_atY.png etc. bu I recommend only one skeleton per video). Don't delete number after *at* - it is start frame of skeleton needed for proper synchronization of annotations with video.
4. Edit keypoints:
    * `python fillKeypoints example/poseX/ example/pose/poseXatY.p`
    * press *Skip* button to find next incomplete frame or go frame by frame using *Next* button.
    * press *Save* to save changes into file.
    * your file is writen as `poseXatY_f.p` (you can rename it to previous version file, but you will loose original annotations).
5. Create long images from annotations:
    * `python proceedVideo.py example/poseX/ -a example/pose/poseXatY_f.p -p example/pose`
    * you should receive one *.png* file `s0.png`. Rename it to `poseX.png`.
6. Repeat steps 1-5 to create another skeleton for your pose.
    * After that you should have bunch of images in `/data/images/example/pose/` folder named `pose0.png pose1.png` etc.
7. Create small images for training from created long images:
    * `python augument.py example/pose -o example/pose_Z` where *Z* is code of pose eg. for stand it's 0.
    * If you didn't type -o argument your short images will be stored in `/data/images/example/pose/pose_aug/` folder.
8. Repeat steps 1-7 for different poses.
    * After that you should have in your `/data/images/example/` folder bunch of folders named `stand_0 sit_1` etc.
9. Create dataset from all images:
    * `python createDataset.py example -d datasetName`.


#### recordVideo.py
Module records stream of RGBD camera and writes output to *.oni* file. Output video will be stored in `/data/videos`.

Usage: `python recordVideo.py -v video_name`:
* *video_name* - name of output video without extinction.

#### proceedVideo.py
Module takes recorded video in *.oni* or regular video format, estimates human skeletons and converts this skeletons into images.
Module also shows video with estimated skeletons.

Usage: `python proceedVideo.py video_path -p proceed -a annotations -v -w -l -k`:
* video_path - path to your video relative to running folder or to `/data/video` folder.
* proceed - proceed mode. Select this option when you want to code estimated skeletons to images. *Proceed* is name of folder where images will be saved (relative to `/data/images/` folder).
* annotations - skeletons will not be estimated. It will be loaded from annotation file with pickle extinction (*.p*) instead.
* -v - view mode. Select this option when you only want to view your video (without estimating skeletons. Useful for viewing *.oni* files or image sequences.
* -w - name of output video (if you want to save proceeded video). Useful to know which coded skeleton images correspond to which frame in proceeded video.
* -l - skeletons will be written into one long image instead of multiple small images for every frame. It can be easily proceeded with *augment* module.
* -k - skeleton annotations will be written into pickle file as absolute instead of encoding it to images. It can be then edited with *fillKeypoints.py* module.

#### fillKeypoints.py
Module is a simple keypoints annotations editor. Keypoints, which are not detected can be easily dragged to it's proper position.
Corrected annotations will be stored in file with input name + '_f'.

Usage: `python fillKeypoints.y video annotations`:
* video - path to video you want to open (it is independent of the keypoint annotations, so please make sure you typed correct path).
* annotations - path to file with keypoints annotations you want to correct.

Buttons:
* *Next* - next frame will be shown. Changes from previous will be saved (but not in the file yet).
* *Previous* - previous frame will be shown. Changes from previous frame will be saved (but not in the file yet).
* *Skip* - skips to next frame where not all keypoints are detected.
* *Save* - saves all made changes into file.
* *Reset* - resets all keypoints positions in current frame.

Point colors:
* *Green* - keypoints detected in annotations file (you can still drag it).
* *Red* - keypoint not detected in current frame (it appears on place from previous frame).
* *Purple* - keypoints has been edited. It will be saved after going to next frame.
On the down left corner of window you can see name of currently edited keypoint.
 

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
