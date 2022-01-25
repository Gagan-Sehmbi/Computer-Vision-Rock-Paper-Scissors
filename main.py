# %%
import cv2
from keras.models import load_model
import numpy as np
import random

# %%
model = load_model('rps_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# %%
with open('labels.txt') as f:
    lines = f.readlines()
    f.close()

options = [(lambda x: x[2:-1])(x) for x in lines]
options

# %%
i = 0
while True: 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)
    # Press q to close the window
    i += 1
    if i == 100:
        i = 0
        opponent = options[random.randint(0,2)]
        me = options[np.argmax(prediction)]
        if opponent == me:
            print(f'Opponent chose {opponent}')
            print(f'I chose {me}')
            print('Result was a draw')
        elif (opponent == 'Rock' and me == 'Paper') or (opponent == 'Paper' and me == 'Scissors') or (opponent == 'Scissors' and me == 'Rock'):
            print(f'Opponent chose {opponent}')
            print(f'I chose {me}')
            print('I win!')
        elif me == 'None':
            print('Not a valid option')
        else:
            print(f'Opponent chose {opponent}')
            print(f'I chose {me}')
            print('I lost')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

