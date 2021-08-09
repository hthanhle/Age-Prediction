"""
UOW, Mon Dec 28 14:37:42 2020
Dependencies: torch>=1.1, torchvision>=0.3.0
"""
import cv2
from dsfd.detect import DSFDDetector
from timeit import default_timer as timer

# In[1]: Load the pre-trained DSFD
weight_path = "./weights/WIDERFace_DSFD_RES152.pth"
detector    = DSFDDetector(weight_path)
print('Loading the model done')

# In[2]: Read an input image
filename = 'worlds-largest-selfie.jpg'
image    = cv2.imread('worlds-largest-selfie.jpg')

# In[3]: Perform face detection
start = timer()
detections = detector.detect_face(image, confidence_threshold=.5, 
                                  nms_iou_threshold=0.3)
end = timer()
num_faces = detections.shape[0]
print('Total detected faces: {}. Detection time: {}'.format(num_faces, end-start))

# In[4]: Visualize the results
for k in range(num_faces): # process each detected face   
    # Draw the bounding box
    cv2.rectangle(image,
                  (detections[k][0], detections[k][1]),
                  (detections[k][2], detections[k][3]),
                  (0,0,255), 2)
    
# Save the output image
cv2.imwrite('out_' + filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))