# pose2pose-demo

This is a pix2pix demo that learns from pose and translates this into a human. A webcam-enabled application is also provided that translates your pose to the trained pose.

## Getting Started

#### 1. Prepare Environment

```
# Clone this repo
git clone git@github.com:GordonRen/pose2pose.git

# Create the conda environment from file
conda env create -f environment.yml
```
#### 2. Configure PyOpenPose

```
https://github.com/FORTH-ModelBasedTracker/PyOpenPose
```
#### 3. Generate Training Data

```
python generate_train_data.py --file Panama.mp4
```

Input:

- `file` is the name of the video file from which you want to create the data set.

Output:

- Two folders `original` and `landmarks` will be created.

If you want to download my dataset, here is also the [video file](https://dl.dropboxusercontent.com/s/isz4tdkfopebwpw/Panama.mp4) that I used and the generated [training dataset](https://dl.dropboxusercontent.com/s/7qjye3efr0czux8/dataset_pose.zip) (1427 images already split into training and validation).

#### 4. Train Model
```
# Clone the repo from Christopher Hesse's pix2pix TensorFlow implementation
git clone https://github.com/affinelayer/pix2pix-tensorflow.git

# Move the original and landmarks folder into the pix2pix-tensorflow folder
mv pose2pose/landmarks pose2pose/original pix2pix-tensorflow/photos_pose

# Go into the pix2pix-tensorflow folder
cd pix2pix-tensorflow/

# Reset to april version
git reset --hard d6f8e4ce00a1fd7a96a72ed17366bfcb207882c7

# Resize original images
python tools/process.py \
  --input_dir photos_pose/original \
  --operation resize \
  --output_dir photos_pose/original_resized
  
# Resize landmark images
python tools/process.py \
  --input_dir photos_pose/landmarks \
  --operation resize \
  --output_dir photos_pose/landmarks_resized
  
# Combine both resized original and landmark images
python tools/process.py \
  --input_dir photos_pose/landmarks_resized \
  --b_dir photos_pose/original_resized \
  --operation combine \
  --output_dir photos_pose/combined
  
# Split into train/val set
python tools/split.py \
  --dir photos_pose/combined
  
# Train the model on the data
python pix2pix.py \
  --mode train \
  --output_dir pose2pose-model \
  --max_epochs 1000 \
  --input_dir photos_pose/combined/train \
  --which_direction AtoB
```

For more information around training, have a look at Christopher Hesse's [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) implementation.

#### 5. Export Model

1. First, we need to reduce the trained model so that we can use an image tensor as input: 
    ```
    python reduce_model.py --model-input pose2pose-model --model-output pose2pose-reduced-model
    ```
    
    Input:
    
    - `model-input` is the model folder to be imported.
    - `model-output` is the model (reduced) folder to be exported.
    
    Output:
    
    - It returns a reduced model with less weights file size than the original model.

2. Second, we freeze the reduced model to a single file.
    ```
    python freeze_model.py --model-folder pose2pose-reduced-model
    ```

    Input:
    
    - `model-folder` is the model folder of the reduced model.
    
    Output:
    
    - It returns a frozen model file `frozen_model.pb` in the model folder.
    
I have uploaded a pre-trained frozen model [here](https://dl.dropboxusercontent.com/s/piuyhvk2tjftdjh/pose2pose_model_epoch_1000.zip). This model is trained on 1427 images with epoch 1000.
    
#### 6. Run Demo

```
python pose2pose.py --source 0 --show 2 --tf-model pose2pose-reduced-model/frozen_model.pb
```

Input:

- `source` is the device index of the camera (default=0).
- `show` is an option to display: 0 shows the normal input; 1 shows the pose; 2 shows the normal input and pose (default=2).
- `tf-model` is the frozen model file.

Example:

![example](example.gif)

## Requirements
- [TensorFlow 1.0.0](https://www.tensorflow.org/)
- [PyOpenPose](https://github.com/FORTH-ModelBasedTracker/PyOpenPose)

## Acknowledgments
Kudos to [Christopher Hesse](https://github.com/christopherhesse) for his amazing pix2pix TensorFlow implementation and [Gene Kogan](http://genekogan.com/) for his inspirational workshop. \
Inspired by [Dat Tran](https://github.com/datitran/face2face-demo).

## License
See [LICENSE](LICENSE) for details.
