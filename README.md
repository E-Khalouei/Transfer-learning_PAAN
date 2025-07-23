# Transfer-learning_PAAN

In this project, I converted noisy scientific signals to audio format and used the CNN14 pretrained model from PAAN as a feature extractor.  

![Project Workflow](https://github.com/user-attachments/assets/b0df25c6-f1b1-48e4-9502-d2e4e84e7078)

*Figure: Workflow from noisy signal to audio, leading to audio-based feature extraction using transfer learning.*

![Project Workflow](https://github.com/user-attachments/assets/3c7ff148-6963-4b6d-adc8-b9bd2b61d710)


**Using the PAAN Pretrained Models**
1- Download all files from [the original repository](https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master)

2- **Adjust the file paths** as needed for your environment.

3- **Note:** The primary modification is in inference.py, which has been adapted in this project to serve as a feature extractor.

4- **Generate embedding files** from your input data using the modified script. The size of each embedding is 2048.

5- See `Example.py` for a demonstration of how to use the embedding files to train your own model. 

 
 
 
