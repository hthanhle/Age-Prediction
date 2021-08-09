'''
UOW, 24/12/2020
Dependencies: keras>=2.0.0, opencv-python>=4.1.0
Reference: https://github.com/ipazc/mtcnn
'''

import cv2
from mtcnn import MTCNN
from timeit import default_timer as timer

# In[1]: Create an MTCNN model
detector = MTCNN()

# In[1]: Read an input image
filename = 'test2.jpg'
image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

# In[2]: Perform face detection
start = timer()
result = detector.detect_faces(image)
end = timer()
print('Total detected faces: {}. Detection time: {}'.format(len(result), end-start))

# In[3]: Visualize the result
for k in range(len(result)): # process each detected face
    bounding_box = result[k]['box']
    keypoints = result[k]['keypoints']
    
    # Draw the bounding box
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,0,255), 2)
    
    # Draw the key points
    cv2.circle(image,(keypoints['left_eye']), 2, (0,255,0), 1)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,255,0), 1)
    cv2.circle(image,(keypoints['nose']), 2, (255,0,0), 1)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,155), 1)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,155), 1)
    
# Save the output image
cv2.imwrite('out_' + filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    
