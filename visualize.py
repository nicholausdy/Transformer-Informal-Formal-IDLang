import pandas as pd
import matplotlib.pyplot as plt

filename = "./result/training_log.csv"
output_file = ["./result/training_graph.png", "./result/validation_graph.png"]

def visualize():
    # plot training graph
    df = pd.read_csv(filepath_or_buffer=filename, sep=',', header=0, index_col=0)
    df.plot(xlabel="Epochs", y=["loss", "accuracy"])
    plt.savefig(output_file[0])
    
    # plot validation graph
    df.plot(xlabel="Epochs", y=["val_loss", "val_accuracy"])
    plt.savefig(output_file[1])

visualize()