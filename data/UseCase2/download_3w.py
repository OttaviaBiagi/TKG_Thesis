import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

import kagglehub
path = kagglehub.dataset_download("afrniomelo/3w-dataset")
print("Path to dataset files:", path)
import kagglehub

path = kagglehub.dataset_download("afrniomelo/3w-dataset")
print("Path to dataset files:", path)