# %% [markdown]
# # Inference of Network Structure
# 
# #### Parameters





import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)





# %%
# Changable variables
Model_Name = "LF"
Epochs = 100
Batch_size =  32

# Define the input and output shapes
input_shape = (200, 10)  # Shape of the input data
output_units = 10 * 10  # Number of output units


test_sample_number = int(185)

# %% [markdown]
# ### Import libraries

# %%
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# %% [markdown]
# ### Read train dataset (inputs and labels)

# %%
# Assuming the files are in the current directory. Modify as needed.
directory = "./Train data/phases/"
directory2 ="./Train data/A/"

# Initialize lists to store the inputs and labels
all_inputs = []
all_labels = []

# Loop through each input and label file
for i in range(1, 1000001):  # assuming files are named from 1 to 10000
    input_filename = os.path.join(directory, f"Phase_snapshot_N10_J4.000000_S{i}.dat")
    label_filename = os.path.join(directory2, f"A_{i}.dat")

    # Read the input file
    input_data = np.loadtxt(input_filename)
    all_inputs.append(input_data)

    # Read the label file
    label_data = np.loadtxt(label_filename)
    all_labels.append(label_data)

# Convert the lists to numpy arrays
all_inputs = np.array(all_inputs)  # This will have shape (10000, 200, 29)
all_labels = np.array(all_labels)  # This will have shape (10000, 29, 29)

# You can now use all_inputs and all_labels for your model
print("Inputs shape:", all_inputs.shape)
print("Labels shape:", all_labels.shape)


# %% [markdown]
# ### Split train and validation data 

# %%
# Assuming all_inputs and all_labels are already loaded
# all_inputs shape is (1000, 200, 29)
# all_labels shape is (1000, 29, 29)

# Flatten the labels to match the output shape
flattened_labels = all_labels.reshape(-1, 10 * 10)  # Shape becomes (1000, 841)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(all_inputs, flattened_labels, train_size=0.9, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)


# %% [markdown]
# 
# ### Create NN models

# %%
def create_model(model_name, input_shape, output_units):
    input_layer = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    # Apply cosine and sine functions using Lambda layers
    x = tf.keras.layers.Lambda(lambda x: tf.math.cos(x))(input_layer)
    y = tf.keras.layers.Lambda(lambda x: tf.math.sin(x))(input_layer)
    
    # Concatenate the transformed inputs
    z = tf.keras.layers.Concatenate(axis=-1)([x, y])
    
    if model_name == "FFF":
        z = tf.keras.layers.Flatten()(z)  # Use the original input_layer for consistency
        z = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.tanh)(z)
        z = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.tanh)(z)
    
    elif model_name == "1F":
        z = tf.keras.layers.Conv1D(filters=64, kernel_size=6, activation=tf.keras.activations.tanh)(input_layer)
        z = tf.keras.layers.Flatten()(z)
    
    elif model_name == "2F":
        z = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(input_layer)
        z = tf.keras.layers.Conv2D(filters=64, kernel_size=(6, 3), activation=tf.keras.activations.tanh)(z)
        z = tf.keras.layers.Flatten()(z)
    
    elif model_name == "LF":
        z = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(input_layer)
        z = tf.keras.layers.LSTM(units=1024, return_sequences=True)(z)  # Set return_sequences=True
        z = tf.keras.layers.LSTM(units=512)(z)
    
    elif model_name == "RF":
        z = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(input_layer)
        z = tf.keras.layers.SimpleRNN(units=64)(z)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    output_layer = tf.keras.layers.Dense(units=output_units, activation=tf.keras.activations.sigmoid)(z)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryCrossentropy()])
    return model

# %% [markdown]
# ### Choose a NN model and Train process

# %%
# Select and create the model
model = create_model(Model_Name , input_shape, output_units)   # Replace with create_model_2F, create_model_LF, or create_model_RF

# Train the model with validation data
history = model.fit(X_train, y_train, epochs=Epochs, batch_size=Batch_size, validation_data=(X_val, y_val))


# %%
def plot_training_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(f'./Results/training_history_model_{Model_Name}.jpg', dpi = 200)
    plt.show()

# Plot the training history
plot_training_history(history)

# %% [markdown]
# ### Read test dataset

# %%
# Assuming the files are in the current directory. Modify as needed.
directory = "./Test data/phases/"
directory2 ="./Test data/A/"

# Initialize lists to store the inputs and labels
all_test_inputs = []
all_test_labels = []

# Loop through each input and label file
for i in range(1, 1001):  # assuming files are named from 1 to 100
    input_test_filename = os.path.join(directory, f"Phase_snapshot_N10_J4.000000_S{i}.dat")
    label_test_filename = os.path.join(directory2, f"A_{i}.dat")

    # Read the input file
    input_test_data = np.loadtxt(input_test_filename)
    all_test_inputs.append(input_test_data)

    # Read the label file
    label_test_data = np.loadtxt(label_test_filename)
    all_test_labels.append(label_test_data)

# Convert the lists to numpy arrays
all_test_inputs = np.array(all_test_inputs)  # This will have shape (1000, 200, 29)
all_test_labels = np.array(all_test_labels)  # This will have shape (1000, 29, 29)

# You can now use all_inputs and all_labels for your model
print("test Inputs shape:", all_test_inputs.shape)
print("Test Labels shape:", all_test_labels.shape)

# %% [markdown]
# ### Test process

# %%
# Assuming you have the test input and label data loaded
X_test = all_test_inputs
y_test = all_test_labels

# Evaluate the model on the test data
#test_loss = model.evaluate(X_test, y_test)
#print(f"Test Loss: {test_loss}")

# Use the model to make predictions on the test data
predictions = model.predict(X_test)

# Print or inspect some of the predictions
print("Predictions:")
print(predictions[:5])  # Print the first 5 predictions for inspection

# Save the predicted values to a file
np.savetxt("predicted_values.txt", predictions, fmt='%.5f', delimiter=' ')

print("Predicted values have been saved to 'predicted_values.txt'.")

# %% [markdown]
# ### Make a sample phase pattern

# %%
# Define the colors for the custom cyclic colormap
colors = [
    (1.0, 0.0, 1.0),  # Magenta (255, 0, 255)
    (1.0, 0.0, 0.0),  # Red (255, 0, 0)
    #(1.0, 0.5, 0.0),  # Orange (255, 128, 0)
    (1.0, 1.0, 0.0),  # Yellow (255, 255, 0)
    (0.0, 1.0, 0.0),  # Green (0, 255, 0)
    (0.0, 1.0, 1.0),  # Cyan (0, 255, 255)
    (0.0, 0.0, 1.0),  # Blue (0, 0, 255)
    (1.0, 0.0, 1.0)   # Magenta (255, 0, 255)
]
# Create the custom cyclic colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)



fig = plt.figure()
#_________________________________________________________
# Read one of the input test data
with open(f"./Test data/phases/Phase_snapshot_N10_J4.000000_S{test_sample_number}.dat") as textFile:         
    lines = [line.split() for line in textFile]

lines = np.array(lines, dtype=float).transpose()  # Convert to numpy array and transpose
print(np.min(lines))
N = len(lines[0])
#plot
plt.imshow(lines, cmap=custom_cmap, aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlim(0,1)
plt.xlim(0,N)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False,  labelsize=12, labelcolor='#262626')
plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False,  labelsize=12, labelcolor='#262626')
plt.ylabel('Node (i)',  fontsize=12, labelpad=12)
plt.xlabel('Time (s)',  fontsize=12)
plt.gca().set_ylim(plt.gca().get_ylim()[::-1])# Reverse the y-axis direction
#_________________________________________________________
plt.gcf().set_size_inches(6, 4)         
plt.savefig(f'./Results/Phase_snapshot_N10_J4.000000_S{test_sample_number}.png', dpi=300)

# %% [markdown]
# ### Make mouse visual cortex adjacency matrix (actual matrix)

# %%
#ax_1 = plt.subplot(3, 4, (1,4))

# Load the matrix from the text file
matrix = np.loadtxt('/home/Najmeh/test_10^4/Test data/A/mouse visual cortex.txt')


# Plot the matrix with reversed colors (white for links, black for no links)
plt.imshow(matrix, cmap='gray_r', interpolation='none' , vmin=0, vmax=1)
plt.colorbar()

# Set axis labels
plt.xlabel('Node(i)')
plt.ylabel('Node(i)')
# Reverse the y-axis
plt.gca().invert_yaxis()

# Save the plot as a PNG file
plt.savefig('mouse visual cortex_adj.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# %%

# Load the data from the text file
data = np.loadtxt('/home/Najmeh/test_10^4/Test data/A/average of predicted values.txt')


# Ensure the data contains exactly 841 values
if data.size != 841:
    raise ValueError("The data does not contain 841 values.")

# Reshape the data into a 29x29 matrix
matrix = data.reshape((29, 29))
print(matrix)



# Plot the matrix with reversed colors (white for links, black for no links)
plt.imshow(matrix, cmap='gray_r', interpolation='none', vmin=0, vmax=1)
plt.colorbar()

# Set axis labels
plt.xlabel('Nodes')
plt.ylabel('Nodes')
# Reverse the y-axis
plt.gca().invert_yaxis()

# Save the plot as a PNG file
plt.savefig('predicted_adj.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the text file
data = np.loadtxt('/home/Najmeh/test_10^4/Test data/A/average of predicted values.txt')

# Ensure the data contains exactly 841 values
if data.size != 841:
    raise ValueError("The data does not contain 841 values.")

# Reshape the data into a 29x29 matrix
matrix = data.reshape((29, 29))

# Round the values: less than 0.5 -> 0, 0.5 or more -> 1
matrix = np.where(matrix < 0.5, 0, 1)
print(matrix)

# Count the number of 1's in the matrix
num_ones = np.sum(matrix == 1)
print(f"Number of 1's in the matrix: {num_ones}")


# Plot the matrix with reversed colors (white for links, black for no links)
plt.imshow(matrix, cmap='gray_r', interpolation='none', vmin=0, vmax=1)
plt.colorbar()

# Set axis labels
plt.xlabel('Nodes')
plt.ylabel('Nodes')

# Reverse the y-axis
plt.gca().invert_yaxis()

# Save the plot as a PNG file
plt.savefig('rounded_adj.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the actual and predicted matrices (both are 29x29 matrices)
actual_matrix = np.loadtxt('/home/Najmeh/test_10^4/mouse visual cortex.txt')
predicted_matrix = np.loadtxt('/home/Najmeh/test_10^4/average of predicted values.txt')


# Reshape the data into a 29x29 matrix
predicted_matrix = predicted_matrix.reshape((29, 29))

# Round the values: less than 0.5 -> 0, 0.5 or more -> 1
rounded_matrix = np.where(predicted_matrix < 0.5, 0, 1)


# Ensure both matrices have the correct shape (29x29)
if actual_matrix.shape != (29, 29) or rounded_matrix.shape != (29, 29):
    raise ValueError("Both matrices must be 29x29.")


# Create a new matrix to hold the color codes
# 0: blue, 1: green, 2: yellow, 3: red
new_matrix = np.zeros_like(actual_matrix, dtype=int)

# Assign values based on the conditions
new_matrix[(actual_matrix == 1) & (rounded_matrix == 1)] = 1 #1  # green
new_matrix[(actual_matrix == 1) & (rounded_matrix == 0)] = 2 #3  # red
new_matrix[(actual_matrix == 0) & (rounded_matrix == 1)] = 2 #2  # yellow
new_matrix[(actual_matrix == 0) & (rounded_matrix == 0)] = 1 #0  # blue

#print(new_matrix)

# Define the custom colormap
cmap = mcolors.ListedColormap(['blue', 'green', 'yellow', 'red'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the matrix
plt.imshow(new_matrix, cmap=cmap, norm=norm, interpolation='none')
#plt.colorbar(ticks=[0, 1, 2, 3], label='Comparison')
plt.clim(-0.5, 3.5)

# Set axis labels
plt.xlabel('Node(i)')
plt.ylabel('Node(i)')

# Reverse the y-axis if needed
plt.gca().invert_yaxis()

# Save and show the plot
plt.savefig('comparison_matrix.png', dpi=300)
plt.show()


# %% [markdown]
# ### Result

# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the actual and predicted matrices (both are 29x29 matrices)
actual_matrix = np.loadtxt('/home/Najmeh/test_10^4/mouse visual cortex.txt')
predicted_matrix = np.loadtxt('/home/Najmeh/test_10^4/average of predicted values.txt')

# Reshape the predicted matrix into a 29x29 matrix
predicted_matrix = predicted_matrix.reshape((29, 29))

# Round the predicted matrix: less than 0.5 -> 0, 0.5 or more -> 1
rounded_matrix = np.where(predicted_matrix < 0.18, 0, 1)

# Ensure both matrices have the correct shape (29x29)
if actual_matrix.shape != (29, 29) or rounded_matrix.shape != (29, 29):
    raise ValueError("Both matrices must be 29x29.")

# Create a new matrix to hold the color codes for the comparison matrix
# 0: blue, 1: green, 2: yellow, 3: red
new_matrix = np.zeros_like(actual_matrix, dtype=int)

# Assign values based on the conditions
new_matrix[(actual_matrix == 1) & (rounded_matrix == 1)] = 1  # green
new_matrix[(actual_matrix == 1) & (rounded_matrix == 0)] = 3  # red
new_matrix[(actual_matrix == 0) & (rounded_matrix == 1)] = 2  # yellow
new_matrix[(actual_matrix == 0) & (rounded_matrix == 0)] = 0  # blue

# Define the custom colormap for the comparison matrix
cmap = mcolors.ListedColormap(['blue', 'green', 'yellow', 'red'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)




# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# Plot the actual matrix
axs[0,0].imshow(actual_matrix, cmap='gray_r', interpolation='none', vmin=0, vmax=1)
axs[0,0].set_title('Actual Matrix')
axs[0,0].set_xlabel('Nodes')
axs[0,0].set_ylabel('Nodes')
axs[0,0].invert_yaxis()

# Plot the predicted matrix
axs[0,1].imshow(predicted_matrix, cmap='gray_r', interpolation='none', vmin=0, vmax=1)
axs[0,1].set_title('Predicted Matrix')
axs[0,1].set_xlabel('Nodes')
axs[0,1].set_ylabel('Nodes')
axs[0,1].invert_yaxis()

# Plot the rounded matrix
axs[1,0].imshow(rounded_matrix, cmap='gray_r', interpolation='none', vmin=0, vmax=1)
axs[1,0].set_title('Rounded Matrix')
axs[1,0].set_xlabel('Nodes')
axs[1,0].set_ylabel('Nodes')
axs[1,0].invert_yaxis()

# Plot the comparison matrix
im = axs[1,1].imshow(new_matrix, cmap=cmap, norm=norm, interpolation='none')
axs[1,1].set_title('Comparison Matrix')
axs[1,1].set_xlabel('Nodes')
axs[1,1].set_ylabel('Nodes')
axs[1,1].invert_yaxis()

# Add a colorbar to the comparison matrix
fig.colorbar(im, ax=axs[1,1], ticks=[0, 1, 2, 3])

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('matrix_comparison.png', dpi=300)
plt.show()


# %%


# %%


# %%

