import sys
sys.path.append('..')

from Summarizer import file_text, summary_
from Translator import translate_1, translate_2
from QnA import QnA_
from Spell_corrector import check_, extract_vocabs_
from Generator import generator_, model_
from text_to_hand import text_to_hw
import streamlit as st
import pandas as pd
from PIL import Image

import re

st.set_page_config(page_title='worKINGS', layout = 'wide', initial_sidebar_state = 'auto')

nav = st.sidebar.radio('NAVIGATION', ['Home', 'Machine Learning', 'Language Translator', 'Text Summarizer', 'Text Generator', 'Text to Handwritten', 'Question & Answering', 'Spell Corrector',])

if nav == 'Home':
    st.markdown('<p style="text-align:center;font-size:6\70px;color:#025954;border-radius:2%;"><b>{}</b></p>'.format('WELL-COME'), unsafe_allow_html=True)
    st.balloons()
    
    st.text("")
    text = "In the chase of talent hunt, we are piling up the place with Intelligence. Making up the boxes of hands-on experiments, TreAsuRe app comes into action. This workspace is a combined shot of both the authors and their growing talent & efforts. It is a space of Machine Learning, Natural Language Processing, and Deep Learning hands-on projects that have been built and the counting is still on."
    st.markdown(f'<p style="text-align:left;">{text}</p>'.format(text), unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.markdown('<p style="text-align:center;color:#6CBBB2;font-size:50px;border-radius:2%;"><b>{}</b></p>'.format('"Open the Navigation,'), unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#6CBBB2;font-size:50px;border-radius:2%;"><b>{}</b></p>'.format('Enjoy the Creation"'), unsafe_allow_html=True)
    
elif nav =='Text to Handwritten':
    st.title('TeXt tO haNDwriTTeN')
    st.text("")
    st.markdown("It is a tool to convert your typed text to handwritten material. You can use it with your assignment in order to convert your text for the submission in handwrittem format.")
    typed_input = st.text_area('Input your text below')
    if st.button("Convert"):
        hw_text = text_to_hw(typed_input)
        st.text('')
        st.text('')
        st.image('Handwritten_text.png')
      
elif nav=='Language Translator':
    st.title('LanGuaGe TrAnsLaToR ToOl')
    st.text("")
    st.markdown("This translator helps you translate text from any language to any language. The translation is done in pretty good manner with satisfying results and appropriate meaning of text without altering the meaning.")
    input = st.text_area('Input your text below')
    
    keys_,lang_names = translate_1()
    selected_lang = st.selectbox('Select language',lang_names)
    to_convert = keys_[lang_names.index(selected_lang)]
    st.text("")   
    if st.button('Run'):
        result = translate_2(input,to_convert)
        import time
        st.text("")
        st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-color: #6CBBB2;
            }
        </style>""",
        unsafe_allow_html=True,
        )
        progress = st.progress(0)

        for i in range(101):
            progress.progress(i)
            time.sleep(0.01)
        
        st.text("")
        st.text("Your translated text...")
        st.success(result)
            
elif nav=='Text Summarizer':
    st.title('TeXt SuMMarIzEr ToOl')
    st.text("")
    st.markdown('The perfectly trained text summarizer helps you summarize your long articles, content, definitions. This will give you concise and meaningful text, in a few lines, containing the most of your content. So, no need to worry about reading long stuffs. Enjoy the summary!')
    st.text('')
    st.text('')
    input2 = st.text_area('Input your text below')
    st.markdown('<center> {} </center>'.format('OR'), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload text File",type=['txt'])       
    st.text('')
    num_sent = st.number_input("Select number of sentences", max_value=50, step=1)
    if st.button('Summarize') :
        st.warning('Please wait, the model is loading. It may take some time.')
        if input:
            final = input2
        if uploaded_file:
            final = file_text(uploaded_file)
        result = summary_(final, num_sent)     
        result = re.sub('[[\d+]*]','', result)  
        st.text("")
        st.text("Your Summary text...")
        st.success(result)

elif nav == 'Question & Answering':
    st.title('QueStioNiNg & AnSweRiNg SysTeM')
    st.text("")
    st.markdown('')
    st.markdown('Steps to follow:')
    st.markdown('* Give the context, mainly the category of question.')
    st.markdown('* Enter the question you want to ask.')
    st.markdown('* Get the answer.')
    st.markdown('<b> {} </b> {}'.format('Note: ', 'You need to provide the context once, and can ask numerous questions from context provided.'), unsafe_allow_html=True)
    st.text("")
    st.text("")
    input = st.text_area('Enter the questioning area')  
    text = generator_(input) 
    st.text("")
    st.text("") 
    ques = st.text_area('Ask your question here!')  
    if st.button('Generate Answer'):
        st.warning('Please wait, the model is loading. It may take some time.')
        res=QnA_(ques, text)
        st.text('')
        st.text('')
        st.text("Your answer...")
        st.write(res)
            
elif nav=='Text Generator':
    st.title('TeXt GeNeRaToR ToOl')
    st.text("")
    st.markdown('The perfectly trained text summarizer helps you summarize your long articles, content, definitions. This will give you concise and meaningful text, in a few lines, contanining the most of your content. So, no need to worry about reading long stuffs. Enjoy the summary!')
    st.text("")
    st.text("")
    input_text = st.text_area('Enter your context') 
    st.markdown('<center> {} </center>'.format('OR'), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload text File",type=['txt'])    
    final=''
    if input_text:
        final = generator_(input_text)
    elif uploaded_file:
        final = file_text(uploaded_file)  
    st.text('')
    st.text('')
    seed_text = st.text_area('Enter the beginning of text')
    st.text('')
    num_words = st.number_input("Select number of words you want to generate", max_value=100, step=1)

    if st.button('Generate text'):
        st.warning('Please wait, the model is loading. It may take some time.')
        result = model_(final, seed_text, num_words)
        st.text("")
        st.text("")
        st.text("Your Generated text...")
        st.form()
        st.success(result)
        
elif nav == 'Spell Corrector':
    st.title('SpeLLinG CoRRecTiOn ToOl')
    st.text("")
    st.markdown('')
    st.markdown('Steps to follow:')
    st.markdown('* Enter the mispelledd word you want to check.')
    st.markdown('* Get the possible corrected words with probability.')
    st.text("")
    st.text("")
    text = generator_('vocabulary')  
    print(text)
    print('--------------------------------')
    vb = extract_vocabs_(text) 
    print(vb)
    st.text("")
    mis_word = st.text_input('Enter the mispelled word')  
    if st.button('Correct'):
        
        res=check_(mis_word, vb)
        st.text('')
        st.text('')
        if len(res)==0:
            st.warning('Sorry! Unable to correct it.') 
        else:
            st.text("Your most probable correct words...")
            for i in res:
                st.success('Corrected word: {}  \n Score: {}'.format(i[0], i[1]))
        
elif nav == 'Machine Learning':
    st.title('MaChINe LeArNIng')
    st.markdown("This is the collection of Machine Learning models over different datasets. The page generates a brief report of models on particular dataset along with the type of algorithms applied on it. The report carries models performance some of which are tuned using hyperparameter tuning. The performance report is given for the models on which all the necessary steps are applied such as Data cleaning, Data Exploration, Feature selection and reduction, etc. The dataset are of two types; Prediction and Classification.")
    col1, col2, col3 = st.beta_columns([3,6,2])
    with col1:
        st.write("      ")

    with col2:
        st.image('img\ml.png')

    with col3:
        st.write("   ")
    
    
    csv_datasets = ['None', 'Air Quality', 'Digits', 'Email Ham-Spam', 'Fetal Health', 'Heart Failure', 'In-vehicle-coupon-recommendation', 'Letter Recognition', 'Naval Propulsion', 'Rice(Gonen & Jasmine)', 'Seismic bumps']
    img_datasets = ['None', 'Brain Tumour', 'Cheetah, Hyena, Jaguar, and Tiger', 'Fruits 360', 'Medical Mnist', 'Pokemon']
    
    about_csv_data = {'Air Quality': 'This dataset considers various features that affect the quality of air in the atmosphere. this dataset has many null values which are filled to make data relevant. Dataset is basically tells the air quality in the city Mumbai. Target feature also has many missing values which are also needed to be operated. There are many aspects that are needed to be considered in cities like Mumbai so as to prevent the degradation of air quality.',
                  'Digits': 'This is a dataset of various digits ranges from 0-9. The data contains .txt format files having array format informations about the digits. Each text file represents a particular digit and inside a text file, various numeric values are given forming a particular digit',
                  'Fetal Health': 'Fetal health depends on many things taken from cardiogram tests such as uterine contraction, fetal movement, light deceleration, etc. This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes: Normal, Suspect, Pathological',
                  'Email Ham-Spam': 'It is inspired from real world problems and hence a real world dataset. It is a dataset containing various email samples that we get in daily life. The fact is, the emails are, now-a-days, not completely secure. There are some categories of emails; ham and spam. Spam emails are irrelevant and are usually scam or illegal/not worthy whereas ham represents the usefull category. the emails are needed to be classified to their particular category.',
                  'Heart Failure': 'Heart failure is a major concern of all time because it is a sudden event. It has a thousand number of samples and various attributes such as hypertension, diabetes, mental health, pacemaker, history of stroke and other various ailments. It has 31 attributes among which 30 attributes described patients’ personal details as well as any kind of diseases they have  gone through in past time, clinically tested by doctors. The measures are capable of suggesting whether the person will survive or not.',
                  'In-vehicle-coupon-recommendation': 'A coupon recommendation dataset is containing   several features(including customer/drivers’ family status, his interests, present occupation or earnings, etc) to classify that whatever the coupon is recommended to the person(who is driver) will be accepted by him or not.',
                  'Letter Recognition': 'Letter Recognition depends on many things taken from observations such as dimensional boxes, height, width etc. Letters may belong to different categories on the basis of the emitted structural properties. Some of which, we know, are already in use formally or informally in day-to-day life. The category of particular letter depends on its variable properties that are very well inscribed in the form of features in a sample dataset that are helpful in categorizing the letters according to their type. In our dataset, features like xbox, ybox, width, high, etc. help in understanding it well. The categories of letter in the Letter Recognition dataset are 26 in number that states that the dataset contains the samples with 26 unique categories of letter.',
                  'Naval Propulsion': 'The experiments have been carried out by means of a numerical simulator of a naval vessel (Frigate) characterized by a Gas Turbine (GT) propulsion plant. The different blocks forming the complete simulator (Propeller, Hull, GT, Gear Box and Controller) have been developed and fine tuned over the year on several similar real propulsion plants. In view of these observations the available data are in agreement with a possible real vessel. In this release of the simulator it is also possible to take into account the performance decay over time of the GT components such as GT compressor and turbines. From various possible outcomes, the task is to calculate the GT turbine coefficients of different observations taken in different scenarios. ',
                  'Rice(Gonen & Jasmine)': 'Dataset has been designed to classify the two varieties of rice seed which humans generally intake as food; Jasmine and Gonen. In the dataset taken, Jasmine and Gonen rice seeds differ in many features like area, major axis length, minor axis length, perimeter, etc.',
                  'Seismic bumps': 'There are numerous outcomes of this high energy source. Rough and tough example of the hazard that it can cause is earthquakes. The data is recorded from the Zabrze-Bielszowice coal mine[1] which is located in the area  of Poland. There are a total of 2,584 records, so the data is prominently skewed towards the non hazardous class as these are more in number. Essentially readings of energy and counts of bumps at the time of one work shift are used to classify a hazardous bump during the other shift. Hazardous bump is a seismic process with greater than 10,000 Joules of energy, and a shifting period of about 8 hours. Just consider an example, practically energy of 10,000 Joules will be the sufficient energy which is required to lift 10,000 tomatoes approximately at a height of 1m above the ground. A class represented with 0 as a result signifies a non hazardous bump, while class representing with 1 is a hazardous bump which should always be less.'}
    
    csv_data_shape = {'Air Quality':'(4998, 12)', 'Digits': 'This data contains numerous text files', 'Fetal Health': '(2126,22)', 
                  'Heart Failure': '(1000, 30)', 'In-vehicle-coupon-recommendation': '(12685, 26)', 'Letter Recognition': '(2126, 17)', 
                  'Naval Propulsion': '(11934, 18)', 'Rice(Gonen & Jasmine)': '(18185, 12)', 'Seismic bumps': '(2584, 19)',
                  'Email Ham-Spam': 'This data contains numerous text files'}
    
    about_img_data = {'Brain Tumour' : 'The dataset we are using is “Brain MRI Images for Brain Tumor Detection”. The dataset contains MRI images of the brain to find if the brain has a tumor or not. This dataset contains a total of 253 images. These images are divided in two classes namely: "yes" and "no". "yes" defines that the brain has a tumor and "no" defines that it does not. The readers are recommended to use the latest version of the dataset which can be downloaded from link in the reference section. The images in the dataset are not evenly divided into the classes. There are 155 images in the "yes" group and 98 images in the "no" group. The images in the dataset are not standardized to common size. The images are of different sizes from 200x200 px to 900x900 px. Each image is a colored image with 3 channels (red, green, blue). For inputting these images in our dataset we are first going to resize each image to 224x 224 px in 3 channels. We are using 178 images for training and 75 for validation.',
                      'Cheetah, Hyena, Jaguar, and Tiger': 'Dataset has been picked from the very standard platform i.e. kaggle. Our folders are carrying images of these animals. Each class carries 900 images in a train set as mentioned via kaggle, hence the dataset is divided into 4 classes; Cheetah, Jaguar, Hyena, Tiger. Dataset is carrying different images belonging to mentioned 4 classes. All images size is 400(H) * 400(W) * 3(RGB). The dataset is primarily designed for deep learning task.', 
                      'Fruits 360': 'The ‘fruit 360’ dataset can be downloaded from addresses given in references and if free and open source. The dataset contains 90483 images of fruits and vegetables where our model have used 67692 images for training and validation of the accuracy of the deep learning model. The images in the dataset are 100x 100 pixels with three channels RGB ( red, green, blue). All the pictures of fruits and vegetables are classified into 131 classes. Different varieties of the same fruit are classified into different categories. It has been made using a Logitech C920 camera. The background is white.',
                      'Medical Mnist': 'This dataset is a simple MNIST-style medical images in 64x64 dimension; There were originaly taken from other datasets and processed into such style. There are 58954 medical images belonging to 6 classes. Classes are such as AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT basically belonging to different body parts developing some kind of diseases that are observed using X-Rays images. The dataset is huge with a large number of images in grayscale mode as the images are generated from CT scan.', 
                      'Pokemon': "One of the most famous T.V. show and even considered as the much liked game is pokemon it is a cartoon based character. This dataset is to classify the type of pokemon with its image. Dataset is defined for 150 pokemons like pikachu, charizard, ditto, etc and there are 20-25 images of each pokemon in the dataset. Most (if not all) of the images have relatively high quality (correct labels, centered). The images don't have extremely high resolutions but are centered amazingly so it's perfect for some light deep learning."}
    
    img_data_shape = {'Brain Tumour': '253', 'Cheetah, Hyena, Jaguar, and Tiger': '4000', 'Fruits 360': '90483', 'Medical Mnist': '58954', 'Pokemon': '150 classes; 20-25 images of each class'} 
    
    st.markdown(f'<p style="color:black;font-size:20px;border-radius:2%;">{"What kind of dataset would you like to choose?"}</p>', unsafe_allow_html=True)
    data_cat = st.radio('', ['None', 'Regression/Classification', 'Image Classification'])
    st.text(" ")
    st.text(" ")
    if data_cat=='None':
        pass
    elif data_cat=='Regression/Classification':
        selected_data_1 = st.selectbox('Select dataset: ',csv_datasets)
        st.text(" ")
        st.text(" ")
        if selected_data_1=='None':
            pass
        else:
            st.markdown(''' ''')
            txt='ABOUT DATA'
            st.markdown(f'<p style="color:#6CBBB2;font-size:24px;border-radius:2%;">{txt}</p>', unsafe_allow_html=True)
            
            st.write(about_csv_data[selected_data_1])
            st.write('<b>{}</b>'.format('Data Shape: '), csv_data_shape[selected_data_1],unsafe_allow_html=True)
        
            st.text(" ")
            st.text(" ")
            st.markdown('<p style="color:#6CBBB2;font-size:24px;border-radius:2%;">{}</p>'.format("RESULT & OBSERVATIONS"), unsafe_allow_html=True)
            st.text(" ")
        
            if selected_data_1=='Air Quality':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
              
                st.write("* Artificial Neural Network")
                st.write("* Linear Regression")
                st.write("* Random Forest Regressor")
                features = ['Linear Regression',  'Random Forest Regressor', 'Artificial Neural Network']
                res = pd.DataFrame(columns=features, index=['explained variance score', 'mean squared error', 'mean absolute error'])
                res.loc['explained variance score'] = ['0.75', '0.86', '-']
                res.loc['mean squared error'] = ['0.22', '0.11', '-']
                res.loc['mean absolute error'] = ['0.32', '0.21', '-']  
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                st.table(res)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Neural Network plot of training and testing"), unsafe_allow_html=True)
                st.image('img\Air.png', width=360)
                st.text(" ")
                st.text("Training loss: 0.16")
                st.text("Validation loss: 0.05")
                
            elif selected_data_1=='Digits':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                st.write("* Random Forest Classifier")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                st.write("Precision     :   0.98")
                st.write("Recall        :   1.00")
                st.write("F1-score      :   0.99")
                st.markdown('<b>{}</b>{}'.format("Accuracy      :   "," 0.95"), unsafe_allow_html=True)
                st.markdown('<b>{}</b>{}'.format("ROC-AUC Score :   "," 0.99"), unsafe_allow_html=True)
                
            elif selected_data_1=='Fetal Health':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
             
                st.write("* Artificial Neural Network")
                st.write("* K Nearest Neighbors Classifier")
                st.write("* Random Forest Classifier")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ['K Nearest Neighbors',  'Random Forest Classifier']
                res = pd.DataFrame(columns=features, index=['Precision', 'Recall', 'F1-Score', 'Log-loss', 'Accuracy'])
                res.loc['Precision'] = ['0.87', '0.87']
                res.loc['Recall'] = ['0.77', '0.78']
                res.loc['F1-Score'] = ['0.82', '0.82']
                res.loc['Log-loss'] = ['0.23', '0.21']
                res.loc['Accuracy'] = ['0.91', '0.91']      
                st.table(res)
                st.text("")
                st.text(" ")
                features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                data = pd.DataFrame([{features[0]: 0.7865, features[1]: 0.00, features[2]: 0.7755, features[3]: 0.00}], index=['ANN'])
                st.table(data)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Neural Network plot of training and testing"), unsafe_allow_html=True)
                st.image('img\health.png', width=360)
                
            elif selected_data_1=='Email Ham-Spam':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                
                st.write("* Naive-Bayes Classifier(Gaussian NB)")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                st.write("Precision     :   0.90")
                st.write("Recall        :   85.5")
                st.write("F1-score      :   0.86")
                st.markdown('<b>{}</b>{}'.format("Accuracy      :   "," 0.87"), unsafe_allow_html=True)
                st.markdown('<b>{}</b>{}'.format("ROC-AUC Score :   "," 0.85"), unsafe_allow_html=True)
                
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("ROC-AUC Curve"), unsafe_allow_html=True)
                st.image('img\spam.png', width=360,caption="ROC-AUC Curve")
                
            elif selected_data_1=='Heart Failure':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
             
                st.write("* Catboost Classifier")
                st.write("* AdaBoost Classifier")
                st.write("* Support Vectors Classifier")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ['Catboost Classifier',  'AdaBoost Classifier', 'Support Vectors Classifier']
                res = pd.DataFrame(columns=features, index=['Precision', 'Recall', 'F1-Score', 'Log-loss', 'Accuracy'])
                res.loc['Precision'] = ['1.00', '1.00', '1.00']
                res.loc['Recall'] = ['1.00', '1.00', '1.00']
                res.loc['F1-Score'] = ['1.00', '1.00', '1.00']
                res.loc['Log-loss'] = ['9.99', '9.99', '9.99']
                res.loc['Accuracy'] = ['1.00', '1.00', '1.00']
                
                st.table(res)
                st.text("")
                st.text(" ")
                
                features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                #res = pd.DataFrame(columns=features, index=['ANN'])
                data = pd.DataFrame([{features[0]: 0.9720, features[1]: 0.10, features[2]: 1.0000, features[3]: 0.07}], index=['ANN'])
                st.table(data)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Neural Network plot of training and testing"), unsafe_allow_html=True)
                st.image('img\heart.png', width=360)
                
            elif selected_data_1=='In-vehicle-coupon-recommendation':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
            
                st.write("* Decision Tree Classifier")
                st.write("* XGBoost Classifier")
                st.write("* Artificial Neural Network")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ['Decision Tree Classifier',  'XGBoost Classifier']
                train_res = pd.DataFrame(columns=features, index=['ROC-AUC Score', 'Accuracy'])
                train_res.loc['ROC-AUC Score'] = ['0.99', '0.71']
                train_res.loc['Accuracy'] = ['0.98', '0.73']
                
                test_res = pd.DataFrame(columns=features, index=['ROC-AUC Score', 'Accuracy', 'Log-loss'])
                test_res.loc['ROC-AUC Score'] = ['0.68', '0.71']
                test_res.loc['Accuracy'] = ['0.69', '0.72']
                test_res.loc['Log-loss'] = ['10.52', '0.55']
                
                st.text("")
                st.text("Training score")
                st.table(train_res)
                st.text("")
                st.text("")
                st.text("Testing score")
                st.table(test_res)
                st.text(" ")
                
                features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                #res = pd.DataFrame(columns=features, index=['ANN'])
                data = pd.DataFrame([{features[0]: 0.6415, features[1]: 0.6333, features[2]: 0.6177, features[3]: 0.6479}], index=['ANN'])
                st.text("Artificial Neural Network")
                st.table(data)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Neural Network plot of training and testing"), unsafe_allow_html=True)
                st.image('img\coup.png', width=360)
                
            elif selected_data_1=='Letter Recognition':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
       
                st.write("* Artificial Neural Network")
                st.write("* Decision Tree Classifier")
                st.write("* Random Forest Classifier")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ['Decision Tree Classifier',  'Random Forest Classifier']
                res = pd.DataFrame(columns=features, index=['Precision', 'Recall', 'F1-Score', 'Log-loss', 'Accuracy'])
                res.loc['Precision'] = ['0.86', '0.96']
                res.loc['Recall'] = ['0.88', '0.98']
                res.loc['F1-Score'] = ['0.87', '0.96']
                res.loc['Log-loss'] = ['4.65', '0.31']
                res.loc['Accuracy'] = ['0.87', '0.96']
                
                st.table(res)
                st.text("")
                st.text(" ")
                
                features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                #res = pd.DataFrame(columns=features, index=['ANN'])
                data = pd.DataFrame([{features[0]: 0.9619, features[1]: 0.08, features[2]: 0.9322, features[3]: 0.20}], index=['ANN'])
                st.table(data)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Neural Network plot of training and testing"), unsafe_allow_html=True)
                st.image('img\letter.png', width=360)
                
            elif selected_data_1=='Naval Propulsion':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
             
                st.write("* XGBoost Regressor")
                st.write("* Linear Regressor")
                st.write("* Random Forest Regressor")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ["XGBoost Regressor",  "Linear Regressor", "Random Forest Regressor"]
                res = pd.DataFrame(columns=features, index=['explained variance score', 'mean squared error', 'mean absolute error', 'root mean squared error'])
                res.loc['explained variance score'] = ['0.99', '0.94', '0.99']
                res.loc['mean squared error'] = ['4.77', '3.30', '3.46']
                res.loc['mean absolute error'] = ['0.0004', '0.0012', '0.0003']
                res.loc['root mean squared error'] = ['0.0006', '0.0018', '0.0005']
                
                st.table(res)
            
            elif selected_data_1=='Rice(Gonen & Jasmine)':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
               
                st.write("* Artificial Neural Network")
                st.write("* Logistic Regression Classifier")
                st.write("* Support Vector Classifier")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ['Logistic Regression Classifier',  'Support Vector Classifier']
                res = pd.DataFrame(columns=features, index=['Precision', 'Recall', 'F1-Score', 'Log-loss', 'Accuracy'])
                res.loc['Precision'] = ['0.99', '0.90']
                res.loc['Recall'] = ['0.99', '0.88']
                res.loc['F1-Score'] = ['0.99', '0.89']
                res.loc['Log-loss'] = ['0.03', '0.69']
                res.loc['Accuracy'] = ['0.98', '0.89']
                res.loc['ROC-AUC Score'] = ['0.99', '0.63']
                
                st.table(res)
                st.text("")
                st.text(" ")
                
                features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                #res = pd.DataFrame(columns=features, index=['ANN'])
                data = pd.DataFrame([{features[0]: 0.9873, features[1]: 0.0439, features[2]: 0.9868, features[3]: 0.0422}], index=['ANN'])
                st.table(data)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Neural Network plot of training and testing"), unsafe_allow_html=True)
                st.image('img\RC.png', width=360)
                
            elif selected_data_1=='Seismic bumps':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
              
                st.write("* Random Forest Classifier")
                st.write("* Logistic Regression Classifier")
                st.write("* Catboost Classifier")
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                features = ['Random Forest Classifier',  'Logistic Regression Classifier', 'Catboost Classifier']
                res = pd.DataFrame(columns=features, index=['Precision', 'Recall', 'F1-Score', 'Log-loss', 'Accuracy'])
                res.loc['Precision'] = ['0.93', '0.59', '0.97']
                res.loc['Recall'] = ['1.00', '0.50', '0.52']
                res.loc['F1-Score'] = ['0.96', '0.50', '0.52']
                res.loc['Log-loss'] = ['0.45', '0.21', '0.21']
                res.loc['Accuracy'] = ['0.93', '0.93', '0.94']
                
                st.table(res)
    elif data_cat=='Image Classification':
        selected_data_2 = st.selectbox('Select dataset: ',img_datasets)
        st.text(" ")
        st.text(" ")
        if selected_data_2=='None':
            pass
        else:
            st.markdown(''' ''')
            txt='ABOUT DATA'
            st.markdown(f'<p style="color:#6CBBB2;font-size:24px;border-radius:2%;">{txt}</p>', unsafe_allow_html=True)
            
            st.write(about_img_data[selected_data_2])
            st.write('<b>{}</b>'.format('Data Shape: '), img_data_shape[selected_data_2],unsafe_allow_html=True)
        
            st.text(" ")
            st.text(" ")
            st.markdown('<p style="color:#6CBBB2;font-size:24px;border-radius:2%;">{}</p>'.format("RESULT & OBSERVATIONS"), unsafe_allow_html=True)
            st.text(" ")
            
            if selected_data_2=='Brain Tumour':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                
                st.write("* Convolutional Neural Network")
                st.write("* InceptionResNetV2")
                
                features = ['Training Accuracy',  'Training Loss', 'Testing Accuracy',  'Testing Loss', 'Validation Accuracy',  'Validation Loss']
                res = pd.DataFrame(columns=features, index=['Convolutional Neural Network', 'InceptionResNetV2r'])
                res.loc['Convolutional Neural Network'] = ['0.7865', '0.4592', '0.7747', '0.5181', '0.7333', '0.6249']
                res.loc['InceptionResNetV2r'] = ['0.8427', '0.3497', '0.7509', '0.4881', '0.7067', '0.5523']

                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                st.table(res)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Convolutioal Neural Network"), unsafe_allow_html=True)
                st.image('img\Bt1.png', width=360, caption='Neural Network plot of training and validation')
                st.text(" ")
                st.markdown('<b>{}</b>'.format("InceptionResNetV2"), unsafe_allow_html=True)
                st.image('img\Bt2.png', width=360, caption='Neural Network plot of training and validation')
                
                
            elif selected_data_2=='Cheetah, Hyena, Jaguar, and Tiger':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                st.write("* VGG19")
                
                features = ['Training Accuracy',  'Training Loss', 'Testing Accuracy',  'Testing Loss', 'Validation Accuracy',  'Validation Loss']
                res = pd.DataFrame(columns=features, index=['Convolutional Neural Network', 'InceptionResNetV2r'])
                res.loc['VGG19'] = ['0.9941', '0.2956', '0.9825', '0.8632', '0.9750', '1.2495']
              
                st.text(" ")
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("SCORE"), unsafe_allow_html=True)
                st.table(res)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("VGG19 Accuracy"), unsafe_allow_html=True)
                st.image('img\cheetah2.png', width=360, caption='The accuracy at the time of training and validation')
                st.text(" ")
                st.markdown('<b>{}</b>'.format("VGG19 Loss"), unsafe_allow_html=True)
                st.image('img\cheetah1.png', width=360, caption='The loss at the time of training and validation')
                st.text(" ")
                
            elif selected_data_2=='Fruits 360':
                st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                
                st.write("* Convolutional Neural Network")
                
                features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                data = pd.DataFrame([{features[0]: 0.9765, features[1]: 0.0729, features[2]: 0.9601, features[3]: 0.2399}], index=['CNN'])
                st.table(data)
                st.text(" ")
                st.text(" ")
                st.markdown('<b>{}</b>'.format("Convolutional Neural Network Plot"), unsafe_allow_html=True)
                st.image('img\FT.png', width=360, caption='Training and Validation phase of CNN model')


            elif selected_data_2=='Medical Mnist':
                    st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                   
                    st.write("* Convolutional Neural Network")
                    
                    features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                    data = pd.DataFrame([{features[0]: 0.9571, features[1]: 0.1027, features[2]: 0.9627, features[3]: 0.1329}], index=['CNN'])
                    st.table(data)
                    st.text(" ")
                    st.text(" ")
                    st.markdown('<b>{}</b>'.format("Convolutional Neural Network Plot"), unsafe_allow_html=True)
                    st.image('img\mnist1.png', width=360, caption='The accuracy at the time of training and validation')
                    st.text(" ")
                    st.image('img\mnist2.png', width=360, caption='The accuracy at the time of training and validation')

    
            elif selected_data_2=='Pokemon':
                    st.markdown('<p style="color:#6CBBB2;font-size:18px;border-radius:2%;">{}</p>'.format("Algorithms Used:"), unsafe_allow_html=True)
                   
                    st.write("* Convolutional Neural Network")
                    
                    features = ['Training Accuracy',  'Training Loss', 'Validation Accuracy',  'Validation Loss']
                    data = pd.DataFrame([{features[0]: 0.9599, features[1]: 0.1463, features[2]: 0.5812, features[3]: 2.5626}], index=['CNN'])
                    st.table(data)
                    st.text(" ")
                    st.text(" ")
                    st.markdown('<b>{}</b>'.format("Convolutional Neural Network Plot"), unsafe_allow_html=True)
                    st.image('img\pk1.png', width=360, caption='The accuracy at the time of training and validation')
                    st.text(" ")
                    st.image('img\pk2.png', width=360, caption='The loss at the time of training and validation')

    
            
            
