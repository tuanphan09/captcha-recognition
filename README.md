# captcha-recognition
### CREATE DATASET ###
  - Install ImageCaptcha: ```pip3 install captcha ```
  - Create dataset: ```python3 create_dataset.py``` <br /><br />
  ![](data/train/c793781cce93bd838c24243d9d24d396.png) kh1kjs <br />
  ![](data/train/ce6a03f96b3aef267e7564dc425a9c78.png) 8kxqyc <br />
  ![](data/train/d5c9e15d07d31f74009dd0a349836137.png) tjjzbu
### TRAINING ### 
  - Change lr, batch, epoch, data path,... in ```config.py``` 
  - Training: ```python3 training.py```
### TESTING ### 
  - Run command: ```python3 predict.py```
### RESULT ####
  Data was created by myself and was used for a small competition on Kaggle:
  https://www.kaggle.com/c/aif-challenge2/
  - Publict test: 91.25%
  - Private test: 78.66%
