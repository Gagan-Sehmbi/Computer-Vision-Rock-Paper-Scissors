# %%
import cv2
from keras.models import load_model
import numpy as np
import random


# %%
def rps_game_wrapper(func):
    def wrapper():
        me = 0
        oppo = 0
        while (me <= 2) and (oppo<=2):
            score = func()
            me += score[0]
            oppo += score[1]
            print(f'Score so far {me} (me) : {oppo} (Opponent)')
        print(f'Final score is {me} (You): {oppo} (Opponent)')
    return wrapper

# %%
@rps_game_wrapper
def rps_game():
    model = load_model('rps_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    with open('labels.txt') as f:
        lines = f.readlines()
        f.close()
    options = [(lambda x: x[2:-1])(x) for x in lines]
    options
    for i in range(72):
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        font = cv2.FONT_HERSHEY_SIMPLEX
        num = 3-(i+1)//24
        cv2.putText(frame,f'{num}',(150,150), font, 5,(0,0,0),2,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if i == 71:
            opponent = options[random.randint(0,2)]
            me = options[np.argmax(prediction)]
            if opponent == me:
                print(f'Opponent chose {opponent}')
                print(f'I chose {me}')
                print('Result was a draw')
                score = (0,0)
            elif (opponent == 'Rock' and me == 'Paper') or (opponent == 'Paper' and me == 'Scissors') or (opponent == 'Scissors' and me == 'Rock'):
                print(f'Opponent chose {opponent}')
                print(f'I chose {me}')
                print('I win!')
                score = (1,0)
            elif me == 'None':
                print('Not a valid option')
                score = (0,0)
            else:
                print(f'Opponent chose {opponent}')
                print(f'I chose {me}')
                print('I lost')
                score = (0,1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return score


# %%
rps_game()
# %%
