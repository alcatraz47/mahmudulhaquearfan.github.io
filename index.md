# Welcome to the world of Md. Mahmudul Haque a.k.a alcatraz47

**Who am I?: I am a Data Science practitioner and have been researching in this domain for about 3 and half years. I have a little contribution to this field. I am mostly interested in Natural Language Processing though my first job as well as my capstone is on Computer Vision. I have publication on textual NLP and reasearch experience on voice based NLP.**

### Hobby

Listening songs, Reading fiction books, watching motor sports specially Formula 1(Kimi Raikkonen is my favourite driver.... Bwoah! ;) )

### Job Experience
Current:
    #### Working as a Machine Learning Researcher at NybSys Pvt. Ltd. since January 2019.
    #### Working as a Data Engineer at Mumsy AB under Transdev sicne Januray 2020.

### Education 

- BSc. in Computer Science and Engineering from North South University, Dhaka, Bangladesh.
- Major: Artificial Intelligence and Machine Learning.
- Institutional Trail Name: Artificial Intelligence.

CGPA: shush!!!.... I am an NSUer!

### Publication

- **Data Mining Techniques to Categorize Single Paragraph Formed Self Narrated Stories (Paper accepted in ICT4SD, 2020)**

The proposed arrangement of this undertaking incorporates order of the passages utilising its temperament. Every one of the sections is self-described, and the number of words in those self-described passages vary from more than 100 to under 4200. The passages are classified utilising three classifications which are: ''Work Stress", ''Bullying" in both social and digital world, and ultimately ''Sexual Harassment" in public activity and digital world. Artificial neural network paragraph vectors: a distributed bag of words and distributed memory were utilised to get the features of each passage and later on to group them a few information mining strategies were employed, and these are: decision tree, k nearest neighbours, Gaussian naive Bayes, and logistic regression. The exactness of every calculation lied between 70\% to 94\% in the validation set. The best model gave 77.46\% F1 score in test sets.

### Capstone
**(The exiting first experience!)**

- **Rice Disease Detection from Leaves**
In this project, the dataset was taken from Kaggle. It contained about 3000 images and later on, I have increased the dataset using an image generator and enhanced the amount of dataset from 3000 to 8000. There are data of 5 diseased leaves and 1 healthy rice leaf. Later on, normalisation and image segmentation was done on them after contour detection with edge detection. Then the images were fed into an image generator for further data variation. After that, 3 custom CNNs, 2 Resnet32(one with Adam optimiser and another one with RADAM optimise)s were used to predict diseases/healthiness from that dataset. The accuracy was about 83% on custom CNNs and 85% on Resnets. The model was used as a worker using MQTT data/message transferring broker service via an Android application.

Technical Report Link: [ResearchGate](https://www.researchgate.net/publication/336639606_Krishok_-An_IoT_Based_Intelligent_Farming_Solution)

### Internship/Research

**This is the first time I have ever worked with voice based deep learning. A real experience to remember! This time I really played with emotions... of people of course! :P** 
- **Emotion Recognition from Voice Using Deep Learning:** 
It was the first project on voice processing along with Deep Neural Networks for sequential data. The dataset contained 7 classifications: Sad, Angry, Happy, Pleasant Surprise, Fear, Neutral, and Disgust. The dataset was collected from the University of Toronto's database system a.k.a T-space. The dataset was not that noisy and had two actors(one young and another one was old women) for recording. The dataset was gender-biased as both of the speakers were women. Speech processing was used on that dataset using MFCC feature extraction and later on, Mel Spectrogram was used on them to deduct the features. 3 custom CNN, 1 LSTM, and 1 Densenet were used as an algorithm to predict the categories on unseen data. Again the proportion in training, validation and testing was 70-20-10, but this state was interchanged by time to make sure the algorithm was working correctly. Later on, our own voices were given to determine how the algorithms work in the different data distribution. On similar data distribution, the algorithm given 99.36% accuracy beating the current state of the art on this dataset which was 85%. On different distributions, the algorithm is given 42% due to excessive noise and particularly no extra pre-processing used on them.

### Personal Projects

Now my exploration in this domain!

- **Skin Cancer Detection Using Deep Neural Networks(Paper on Process)**
The sole purpose of this project was to detect the type/category of skin cancer from the given pigmented skin lesion. The types are Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and haemorrhage, vasc). The dataset was collected from Kaggle’s Skin Cancer MNIST: HAM10000 dataset. The dataset was split into 3 partitions: Training(70%), Validation(20%), and lastly Testing(10%). At first, the dataset was normalised and then it was segmented according to the area. Then data generator was used to provide more variation during training time in batch. The whole project was done using the following algorithms: Custom CNN, Custom RESNET32, Pre-Trained Resnet50, Pre-Trained Resnet101. The highest accuracy was about 93% in Resnet101.

- **Liveness/Presentation Error Detection**
The objective of this project was to catch whether a/some person(s) is really present in a face recognition system or not. The dataset was collected manually by me and my friends. The dataset was directly fed into custom CNN and one Resnet50 algorithm without further preprocessing except resizing. Accuracy on the training set was about 99%.

- **Bangla Handwritten Digit Recognition**
I have used a dataset from Kaggle and used some image processing before jumping into the project. At first, I used Random Forest Regression Classifier to detect and recognise the digits. Later on, I have used a Convolutional Neural Network to predict on the digits.
Email Spam Classifier using Support Vector Machine(SVM):
Here, I used SVM(linear classifier and Gaussian Kernel) to detect whether an email is a spam or not. For this project,  I used the labelled dataset of Coursera’s Machine Learning program and also borrowed some optimisation algorithm from it.

- **Compressing Image Using Clustering Algorithm**
Here, I used one picture of mine and my friend to compress. At first, I lowered the dimension of it and then used K-means Clustering algorithm into it. Here, I borrowed the optimisation algorithm of Coursera’s Machine Learning program to optimise the parameters of my algorithm.

### Honour and Award

8th at Team Contest of NeurIPS AutoDL challenges (Auto Speech Challenge) co-hosted by Google, Cha-Learn, and 4paradigm.

### Online Profiles

1. LinkedIn: [Md. Mahmudul Haque](https://www.linkedin.com/in/md-mahmudul-haque-8a5484b2)
2. Github: [alcatraz47](https://github.com/alcatraz47?tab=repositories)
3. Facebook: [Mahmudul Haque Arfan](https://www.facebook.com/mahmud.arfan.alcatraz47)

### Some show off... sorry certificates

- Machine Learning Via Coursera Online MOOC from Stanford University: [verification-link](https://www.coursera.org/account/accomplishments/certificate/56GE9TSYS4K2)
- Neural Networks and Deep Learning: [verification-link](https://www.coursera.org/account/accomplishments/certificate/BBQR5LBFE78B)
- Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization: [verification-link](https://www.coursera.org/account/accomplishments/certificate/DUTH4LKFWV87)
- Mathematics for Machine Learning: Linear Algebra Via Coursera Online MOOC from Imperial College London: [verification-link](https://www.coursera.org/account/accomplishments/certificate/DUTH4LKFWV87)

### Reference
DR. MOHAMMAD RASHEDUR RAHMAN
Professor & Graduate Coordinator
PhD in Computer Science, University of Calgary, Canada
MS in Computer Science, University of Manitoba, Canada
BS in Computer Science and Engineering, BUET, Bangladesh.
