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

## Milestone 3 - Create a rock paper scissors game

- Next the rules for the rock paper scissors game is created
- The computer input is randomly generated from a list (requires python random module)

```python
"""
import random
computer = random.choice(['ROCK','PAPER', 'SCISSORS'])
user_prediction = input("Enter your move (ROCK, PAPER or SCISSORS): ")
"""
```

- The rock paper scissors game uses if-else statments to decide the winner

```python
"""
if user_prediction == 'ROCK':
    if computer == 'ROCK':
        print('DRAW')
    elif computer == 'PAPER':
        print('YOU LOSE')
    elif computer == 'SCISSORS':
        print('YOU WIN')
elif user_prediction == 'PAPER':
    if computer == 'ROCK':
        print('YOU WIN')
    elif computer == 'PAPER':
        print('DRAW')
    elif computer == 'SCISSORS':
        print('YOU LOSE')
elif user_prediction == 'SCISSORS':
    if computer == 'ROCK':
        print('YOU LOSE')
    elif computer == 'PAPER':
        print('YOU WIN')
    elif computer == 'SCISSORS':
        print('DRAW') 
else:
    print('TRY AGAIN')
"""
```
- The statements are converted into a function that takes in a user and computer input as arguments
- The function is placed outside the while loop and called within it

```python
"""
def compare_predictions(user_prediction, computer):
    if user_prediction == 'ROCK':
        if computer == 'ROCK':
            return 'DRAW'
        elif computer == 'PAPER':
            return 'YOU LOSE'
        elif computer == 'SCISSORS':
            return 'YOU WIN'
    elif user_prediction == 'PAPER':
        if computer == 'ROCK':
            return 'YOU WIN'
        elif computer == 'PAPER':
            return 'DRAW'
        elif computer == 'SCISSORS':
            return 'YOU LOSE'
    elif user_prediction == 'SCISSORS':
        if computer == 'ROCK':
            return 'YOU LOSE'
        elif computer == 'PAPER':
            return'YOU WIN'
        elif computer == 'SCISSORS':
            return 'DRAW'
    else:
        return 'TRY AGAIN'

play = compare_predictions(user_prediction, computer)
print(play)

"""
```

- Technologies used: python  

>Insert an image or screenshot of what built so far

## Milestone 4 - Use the camera as input for the game

- The next step brings everything together to create a functional game using the camera input and model


- A function is created that will interpret the model prediction
- The model produces an nparray with 4 probabilities corresponding to each trained image label (rock, paper, scissors or none)
- If the probability that the user is displaying 'ROCK' is over 0.5 based on the trained model, the user_prediction is defined as 'ROCK'

```python
"""
def model_predict(prediction):
    if prediction[0][0] > 0.5:
        user_prediction = 'ROCK'
    elif prediction[0][1] > 0.5:
        user_prediction = 'PAPER'
    elif prediction[0][2] > 0.5:
        user_prediction = 'SCISSORS'
    else:
        user_prediction = 'NONE'
    return user_prediction
"""
```
- The functions are used in the while loop with the following code:
- The user prediction is replaced with the processed input from the camera and model 

```python
""" 
    prediction = model.predict(data)
    computer = random.choice(['ROCK','PAPER', 'SCISSORS'])
    user_prediction = model_predict(prediction)
    game_outcome = compare_predictions(user_prediction, computer)
"""
```

```python
"""Insert your code here"""
```

>Insert an image or screenshot of what built so far

## Conclusions

- what you understood about the project, how would you improve it or take it further 
