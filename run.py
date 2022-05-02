from data_loader import DataLoader
from transformer import TransformerModel

import pandas as pd

# data variables
filename = './dataset.csv'
train_proportion = 0.8

# model hyperparameters
embed_dim = 16
num_heads = 2
ff_dim = 16
batch_size = 8
epochs = 70

# saving model
model_name = "./result/saved_model"
csv_file = "./result/training_log.csv"

def pipeline():
    data = DataLoader(filename, train_proportion)

    vocab_size = data.x_train.shape[1]
    x_train_arr = data.x_train.toarray()
    x_val_arr = data.x_val.toarray()

    # train and validate model
    transformer = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim)
    transformer.model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")
    transformer.model.summary()

    history = transformer.model.fit(
        x_train_arr, data.y_train, batch_size, epochs, shuffle=True,
        validation_data=(x_val_arr, data.y_val),
    )

    # save model
    transformer.model.save(model_name)

    # save training log
    history_df = pd.DataFrame(history.history)
    with open(csv_file, mode='w') as f:
        history_df.to_csv(f)

pipeline()
