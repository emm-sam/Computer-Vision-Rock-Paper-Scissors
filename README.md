# Computer-Vision-Rock-Paper-Scissors
In this lab, you will create an interactive Rock-Paper-Scissors game, in which the user can play with the computer using the camera.

> This is a computer vision project using Google's Teachable Machine to create (train) an image based machine learning algorithm to recognise hand movements from the laptop camera input to play the classic game "Rock, Paper, Scissors" against the computer. 
> 

## Milestone 1 - Train the model

- The first step is to train an image based machine learning model using the laptop camera on [Teachable Machine](https://teachablemachine.withgoogle.com/)
  - More images mean a more ?effective model, including different backgrounds, lighting levels, distance from camera etc
  - This was repeated once I realised that the model was not sensitive enough for the game 
- The code is then available to download in Tensorflow format

```python
"""No code to display"""
```

>[Teachable Machine Screenshot](/Desktop/RPS_screenshots/Milestone_1.jpeg)


## Milestone 2 - Run the model locally

- The next step is to get the model running locally
- This involves creating a conda environment "computer_vision"
- Then installing the dependencies: opencv-python, tensorflow, ipykernal within the new environment
- Python code required to run the model provided by AiCore
- Technologies used: Tensorflow, Keras, VSCode 

Creating a new environment
```python
"""
conda create --name computer_vision
conda activate computer_vision
"""
```

Running the model from a local python notebook
```python
"""
import cv2
from keras.models import load_model
import numpy as np
model = load_model('YOUR_MODEL.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)

    # Press q to close the window
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
"""
```

>Insert an image or screenshot of what built so far

## Milestone 3

- Next the rules for the rock paper scissors game is created
- The computer input is randomly generated from a list (requires python random module)
- 


```python
"""
import random
computer = random.choice(['ROCK','PAPER', 'SCISSORS'])
user_prediction = input("Enter your move (ROCK, PAPER or SCISSORS): ")
"""
```

>Insert an image or screenshot of what built so far

## Milestone 4

- What built? What technologies used? Why have you used them?

```python
"""Insert your code here"""
```

>Insert an image or screenshot of what built so far

## Conclusions

- what you understood about the project, how would you improve it or take it further 
