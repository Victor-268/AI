from detecto import core, utils,visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
import numpy as np
model = core.Model.load('model_weights.pth', ['Shoulder-good','Spine-good','Spine-bad','Shoulder-bad','Neck-good', 'Neck-bad'])
image = utils.read_image('Test/test2.png')
predictions = model.predict(image)
labels, boxes, scores = predictions
thresh=0.6
filtered_indices=np.where(scores>thresh)
filtered_scores=scores[filtered_indices]
filtered_boxes=boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(image, filtered_boxes, filtered_labels)

