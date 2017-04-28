# Sign Language Recognition System

In this project, I build a system that can recognize words communicated using the American Sign Language (ASL). I began the project with a preprocessed dataset of tracked hand and nose positions extracted from video. The goal was to train a set of Hidden Markov Models (HMMs) using part of this dataset to try and identify individual words from test sequences.

This project is written in **Python 3** presented in Jupyter Notebook and uses Python libraries: NumPy, SciPy, scikit-learn, pandas, matplotlib and [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

### Additional Information
##### Provided Raw Data

The data in the `asl_recognizer/data/` directory was derived from
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php).
The handpositions (`hand_condensed.csv`) are pulled directly from
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv`
file.
