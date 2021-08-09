# Low-bias Age Prediction
## Descriptions
This project develops an age prediction system consisting of two stages: 

1. Face detection and alignment using i) Retinaface [[1]](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html), MTCNN [[2]](https://ieeexplore.ieee.org/document/7553523), and iii) DSFD [[3]](https://ieeexplore.ieee.org/document/8954268)
2. Age estimation using i) basic regression, ii) CORAL [[4]](https://www.sciencedirect.com/science/article/pii/S016786552030413X), and iii) SSRNet [[5]](https://www.ijcai.org/proceedings/2018/150)

![alt_text](/output/test1_out.jpg)
![alt_text](/output/test2_out.jpg)
![alt_text](/output/test3_out.jpg)

## Quick start
### Installation
1. Install PyTorch=1.2.0 following [the official instructions](https://pytorch.org/)
2. git clone https://github.com/hthanhle/Age-Prediction
3. Install dependencies: `pip install -r requirements.txt`

### Test
Please run the following commands: 

1. To test on a single image: `python get_age.py --input test.jpg --output test_out.jpg --detector retinaface --estimator ssrnet`
2. To test on camera: `python get_age_cam_coral.py`
3. To test on a single video: `python get_age_video_ssrnet.py --input test.mp4 --output test_out.mp4`
