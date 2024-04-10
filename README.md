# Body-Gesture

For Gesture recognition their exist a handful of models and methods, like ML, DL or use pretrained model through libraries. There exist 2 most prominent open-source Libraries,
•	MediaPipe
•	Openpose
We are proceeding with the MediaPipe as there exist more Remarkable performance in body landmarks identifications. To implement it we need to import NumPy, opencv and matplotlib.

About MediaPipe
The MediaPipe Pose Landmarker task lets you detect landmarks of human bodies in an image or video. You can use this task to identify key body locations, analyze posture, and categorize movements. This task uses machine learning (ML) models that work with single images or video. The task outputs body pose landmarks in image coordinates and in 3-dimensional world coordinates.
The Pose Landmarker uses a series of models to predict pose landmarks. The first model detects the presence of human bodies within an image frame, and the second model locates landmarks on the bodies.
•	Pose detection model: detects the presence of bodies with a few key pose landmarks.
•	Pose landmarker model: adds a complete mapping of the pose. The model outputs an estimate of 3-dimensional pose landmarks.
This model uses a convolutional neural network similar to MobileNetV2 and is optimized for on-device, real-time fitness applications. This variant of the BlazePose model uses GHUM, a 3D human shape modeling pipeline, to estimate the full 3D body pose of an individual in images or videos.  

Pose estimation models have extended the pose landmark predictions into 3D space. This additional step normalises the orientation of the subject within the image and allows for more robust downstream analysis of the captured landmarks.

Mediapipe gives us option of three type of pretrained models to run it on our local machines 
 

Loading Model 

To implement the task, we have download package and model externally 
!pip install -q mediapipe (or) pip install mediapipe

U have to install pre trained model, in this assignment I am utilizing Pose landmarker(Heavy), there exist two methods to download model (excluding documentation  method with  Colab)
•	Pasting the below link in browser will install the model:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

after installing the model, we will have to load it by defining the path, after this we can direct follow the implementation defined in documentation of Medaipipe. 

•	Loading the model directly with criteria with library (used in submitted code):
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.3,
    model_complexity=2
)



Basic Implementation of model


We can directly process the image by loading the model (we are utilizing pose landmarker Heavy)


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create an PoseLandmarker object which was obtained by downloading by link 

base_options = 
python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Load the input image.
image = mp.Image.create_from_file(r"women.jpg")

# Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# Process the detection result and visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
plt.imshow(cv2.cvtColor(annotated_image, cv2.IMREAD_COLOR))


 
As we have mentioned that image is landmarked in 3-dimensional space, with which we can exactly print and visualize the human pose.
result = pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

if result.pose_landmarks:
    for i in range(32):
        print(mpPose.PoseLandmark(i).name)
        print(result.pose_landmarks.landmark[mpPose.PoseLandmark(i).value])
imgHeight, imgWidth, _ = image.shape


This will print the exact coordinates of landamarks in space 

	NOSE
	x: 0.33996004
	y: 0.18860686
	z: -0.1549105
	visibility: 0.9999999

	LEFT_EYE_INNER
	x: 0.3465245
	y: 0.1558671
	z: -0.13655749
	visibility: 0.99999976

mpDraw.plot_landmarks(
    result.pose_world_landmarks,
    mpPose.POSE_CONNECTIONS
)
  

This above image is the 3-D visualization of women pose.
Next part how are we going to recognize the pose of the human at the live stream data?

We are going to identify the angles with each consecutive movements of humans and by coordinating with each frame of data. We have obtained angles for specific poses by trail and error methods and lastly, we implemented with if condition.

def calculateAngle(landmark1, landmark2, landmark3):

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    # Return the calculated angle.
    return angle

For identifying angle we implemented above code, by sending three coordinates because we don’t have a concreate plane to measure.

For example:
 
We want to find angle of shoulder which is 12 we will give coordinates of 14, 12 and 24 to the function.


Model Implementation:

Gestures:
•	Hi
•	Dublebicep
•	Still

We implemented on three poses which focuses on three criteria.

•	In motion (Hi)
•	Good form (Dublebicep) 
•	No motion (Still).

By passing frames of data with live camera, model able to recognize and print out on the top right corner.

 


Thank you for your Time
