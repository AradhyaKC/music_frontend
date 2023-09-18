import numpy as np

def index(prediction):
    z=['Blues','Classical','Country','Disco','Hip-Hop','Jazz','Metal','Pop','Reggae','Rock']
    predicted_indices=np.argmax(prediction,axis=1)
    if predicted_indices[0] in predicted_indices[1:]:
        predicted_index=predicted_indices[0]
    elif predicted_indices[1]==predicted_indices[2]:
        predicted_index=predicted_indices[1]
    else:
        predicted_index=predicted_indices[0]
    return z[predicted_index]