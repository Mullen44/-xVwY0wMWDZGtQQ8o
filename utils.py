## Script that contains all useful functions to keep the main notebook looking cleaner

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score 
import pandas as pd
import seaborn as sn
import numpy as np

def resultsReport(actual, predictions) :
    
    # Print F1 Score
    print('F1 Score ', f1_score(actual, predictions))
    
    # Visualize classification report with precision, recall and F1 score
    print(classification_report(actual, predictions))

    # Create Confusion matrix vector
    confMatrix = confusion_matrix(actual, predictions)
    dfConfMatrix = pd.DataFrame(confMatrix, index = [i for i in "01"], 
                                columns = [i for i in "01"])

    # Plot Confusion matrix
    sn.heatmap(dfConfMatrix, annot = True)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
def analyzeAttribute(data, colName) :
    # Display Average
    print("Average " + colName + " value: ", data[colName].mean())
    
    # Plot histogram
    bins = np.arange(1,7) - 0.5
    plt.hist(data[colName], bins)
    plt.title(colName)
    plt.show()
