# Social Media Toxic Comment Classifier

## Project Background and Rationale

For practical purposes, most social media sites have a lot of extremely negative comments, and many of these are 
directed at other users. This toxic atmosphere dissuades people from adopting the social media site due to repeated 
negative interactions with toxic users. 

The purpose of this project is to:
 * Create an NLP model that can accurately detect whether a comment is toxic or not
 * Create an auxillary NLP that can further sub-categorize each toxic comment into one of six sub-classifications: 
    1. General Toxic
    2. Severe Toxic
    3. Obscene
    4. Threat
    5. Insult
    6. Identity-Based Hate.
    
## Data Set Details
The dataset used to train and test the NLP model is available [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

> Disclaimer: the dataset for this project contains text that may be considered profane, vulgar, or offensive.

The NLP model was trained using 474,807 Wikipedia discussion forum comments, of which approximately 10% of which were 
toxic in some way, and the other 90% were benign (nontoxic) comments. The test data set consists of 100,000 different 
Wikipedia discussion forum comments from the same source.

## To run this application from scratch:
1. Download `train.py`, `test.py`, and `corpus_utils.py` and place the 
three in an empty folder somewhere on your machine.
2. Place the corpus in the same folder as the three python files, and 
rename it to `dataset.txt`
3. Create a folder at this directory called `dictionaries` and leave it 
empty.
4. Open a command prompt in the folder with your python files.
5. Run the following command: `pip install nltk`
6. Run the following command: `train.py`. When the program finishes 
executing, you should have many dictionary files and a count file in 
the `dictionaries/` subfolder.
7. Run the following command: `test.py`. When this program finishes 
executing, you will have a new `test_results.txt` file at the same 
location as the .py files, which will contain the test results.

## Notes:
* The `train.py` program takes a long time to write the trigram files. For nearly 
500,000 comments, it took nearly 6 hours to run on my machine. This 
time can be sped up dramatically if you do not write the trigram files.
* The `test.py` is comparatively much faster, taking only 20 minutes to 
complete testing of 100,000 documents on my machine.
* You can easily modify some of the global variables at the top of 
`test.py` and `train.py` to reduce the size of train and test. I recommend 
10,000 for each in order to run in a few minutes with reasonable 
accuracy.