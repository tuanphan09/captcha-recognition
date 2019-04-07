# captcha-recognition
### CREATE DATASET ###
  - Install ImageCaptcha: ```pip3 install captcha ```
  - Create dataset: ```python3 create_dataset.py```
  ![](data/train/c793781cce93bd838c24243d9d24d396.png)
  ![](data/train/ce6a03f96b3aef267e7564dc425a9c78.png)
### TRAINING ### 
  - Change lr, batch, epoch, data path,... in ```config.py``` 
  - Training: ```python3 training.py```
### TESTING ### 
  - Run command: ```python3 predict.py```
### RESULT ####
  Data was created and used for a small competition on Kaggle:
  https://www.kaggle.com/c/aif-challenge2/
  - Publict test: 91.25%
  - Private test: 78.66%
