## **Execution Instructions**
<br>
 
### Option 1: Running on your computer locally
 
To run the notebook on your local system set up a [python](https://www.python.org/) environment. Set up the [jupyter notebook](https://jupyter.org/install) with python or by using [anaconda distribution](https://anaconda.org/anaconda/jupyter). Download the notebook and open a jupyter notebook to run the code on local system.
 
The notebook can also be executed by using [Visual Studio Code](https://code.visualstudio.com/), and [PyCharm](https://www.jetbrains.com/pycharm/).

**Python Version: 3.8.10**

* Create a python environment using the command 'python3 -m venv myenv'.

* Activate the environment by running the command 'myenv\Scripts\activate.bat'.

* Install the requirements using the command 'pip install -r requirements.txt'

* Run engine.py with the command 'python3 engine.py'.
 

 
### Option 2: Executing with Colab
Colab, or "Collaboratory", allows you to write and execute Python in your browser, with access to GPUs free of charge and easy sharing.
 
You can run the code using [Google Colab](https://colab.research.google.com/) by uploading the ipython notebook.

```
lstm_p2
├─ data
│  ├─ airline_sentiment.csv
│  ├─ alice.txt
│  └─ data.zip
├─ engine.py
├─ lib
│  ├─ images
│  │  ├─ decay.png
│  │  ├─ entropy.png
│  │  ├─ example_softmax_1.png
│  │  ├─ lstm.png
│  │  ├─ many-to-one.png
│  │  ├─ many-to-one_detail.png
│  │  ├─ mlp.png
│  │  ├─ rnn.png
│  │  ├─ rnn_animation.gif
│  │  ├─ temp_animation.gif
│  │  ├─ temp_vs_no_temp.png
│  │  └─ types_rnn.png
│  ├─ lstm_p2.ipynb
│  └─ notebook.zip
├─ ml_pipeline
│  ├─ process.py
│  ├─ train.py
│  └─ utils.py
├─ output
│  └─ sentiment_model.h5
├─ Readme.md
├─ requirements.txt
├─ videos
│  └─ Many-to-One LSTM for Sentiment Analysis and Text Generation
│     ├─ 10_Understanding_Text_Generation_And_Bias.mp4
│     ├─ 11_Understanding_Entropy_In_Machine_Learning.mp4
│     ├─ 12_Understanding_Softmax_Temperature.mp4
│     ├─ 13_Wrapping_Up.mp4
│     ├─ 1_Introduction.mp4
│     ├─ 2_Many_To_One_Architecture.mp4
│     ├─ 3_Preprocessing_Data.mp4
│     ├─ 4_Bag_Of_Words_Embeddings.mp4
│     ├─ 5_Training_Lstm_Model_For_Sentiment_Analysis.mp4
│     ├─ 6_Sentiment_Analysis_Model_Results.mp4
│     ├─ 7_Generating_Text_With_Lstms.mp4
│     ├─ 8_Structuring_Data_For_Text_Generation.mp4
│     └─ 9_Training_And_Generating_Text.mp4
└─ videos.zip

```