import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, filename, test_proportion, val_proportion):
        arr_csv = DataLoader.load_csv(filename)
        x, y = DataLoader.create_data(arr_csv)
        x_preproc = DataLoader.preprocess_data(x)
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = \
            DataLoader.split_data(x_preproc, y, test_proportion, val_proportion) 


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
    def split_data(x_preproc,y, test_proportion = 0.2, val_proportion = 0.2):
        # create x_train, y_train, ,x_test, y_test, x_val, y_val
        # test_proportion = proportion of dataset used as test data (if 0.8, it means 80/20 train-test split)
        # val_proportion = proportion of dataset used as validation data from training data (if 0.8, it means 80/20 train-val split)

        #split into test and training dataset
        x_train, x_test, y_train, y_test = train_test_split(x_preproc, y, test_size= test_proportion)

        #split into validation and training dataset
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size= val_proportion)

        return x_train, y_train, x_val, y_val, x_test, y_test
         

# test
# data_loader = DataLoader('./dataset.csv', 0.2, 0.2)
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

# print("X_test")
# print(data_loader.x_test)
# print(data_loader.x_test.shape)

# print("y_test")
# print(data_loader.y_test)
# print(len(data_loader.y_test))
