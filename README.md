# captcha-recognition
### CREATE DATASET ###
  - Install ImageCaptcha: ```pip3 install captcha ```
  - Create dataset: ```python3 create_dataset.py```
### TRAINING ### 
  - Change lr, batch, epoch, data path,... in ```config.py``` 
  - Training: ```python3 training.py```
### TESTING ### 
  - Run command: ```python3 predict.py```
### RESULT ####
Using data from a competition on Kaggle:
https://www.kaggle.com/c/aif-challenge2/
  - Publict test: 91.25%
  - Private test: 78.66%
