from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

# Load the 'accuracy' metric
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Compute the accuracy using the predictions and true labels
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df):
    # Convert the class labels to numpy array for compute_class_weight
    class_labels = np.array(sorted(df['label'].unique()))
    # Calculate class weights
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=class_labels,
                                         y=df['label'].to_numpy())
    return class_weights
