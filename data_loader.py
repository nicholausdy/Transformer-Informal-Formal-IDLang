import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, filename, train_proportion):
        arr_csv = DataLoader.load_csv(filename)
        x, y = DataLoader.create_data(arr_csv)
        x_preproc = DataLoader.preprocess_data(x)
        self.x_train, self.y_train, self.x_val, self.y_val = \
            DataLoader.split_data(x_preproc, y, train_proportion) 


    @staticmethod
    def load_csv(filename):
        # load CSV as numpy array
        arr_csv = np.loadtxt(filename, delimiter=";", skiprows=1 , dtype= str)
        return arr_csv

    @staticmethod
    def create_data(arr_csv):
        # separate CSV into x and y dataset
        x = []
        y = []

        for record in arr_csv:
            x.append(record[0])

            if record[1] == '0':
                y.append(0)
            else:
                y.append(1)

        return np.array(x), np.array(y)

    @staticmethod
    def preprocess_data(x):
        # vectorize text data into a matrix of token counts
        vectorizer = CountVectorizer(ngram_range=(1,1)) # (1,1) means unigram (i,e., single words) only
        x_preproc = vectorizer.fit_transform(x)
        # print(vectorizer.get_feature_names_out())
        return x_preproc
        

    @staticmethod
    def split_data(x_preproc,y,train_proportion=0.8):
        # create x_train, y_train, x_val, y_val
        # train proportion = proportion of dataset used as train data (if 0.8, it means 80/20 test-val split)
        x_train, x_val, y_train, y_val = train_test_split(x_preproc, y, test_size= 1 - train_proportion)
        return x_train, y_train, x_val, y_val
         

# test
# data_loader = DataLoader('./dataset.csv',0.8)
# print("X_train")
# print(data_loader.x_train)
# print(data_loader.x_train.shape)

# print("y_train")
# print(data_loader.y_train)
# print(len(data_loader.y_train))

# print("X_val")
# print(data_loader.x_val)
# print(data_loader.x_val.shape)

# print("Y_val")
# print(data_loader.y_val)
# print(len(data_loader.y_val))
