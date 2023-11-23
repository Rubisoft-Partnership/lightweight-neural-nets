def mnist_csv_to_custom_format(csv_filepath, output_filepath):
    """
    Converts the MNIST dataset from CSV format to the desired custom format.
    
    csv_filepath: path to the CSV file containing the MNIST dataset.
    output_filepath: path where the converted file will be saved.
    """
    with open(csv_filepath, 'r') as csv_file, open(output_filepath, 'w') as output_file:
        for line in csv_file:
            # Split the string into a list of values
            values = line.strip().split(',')

            # Get the class label (first value)
            class_label = int(values[0])

            # Prepare class labels for 10 classes (0 to 9)
            class_labels = ['0']*10
            class_labels[class_label] = '1'

            # Prepare the feature values, normalizing them from 0-255 to 0.0-1.0
            features = [f'{int(v)/255:.4f}' for v in values[1:]]

            # Concatenate the features and the class labels
            converted_line = ' '.join(features + class_labels) + '\n'
            
            # Write the converted line to the output file
            output_file.write(converted_line)

# Run the function to perform the conversion
mnist_csv_to_custom_format("mnist_test.csv", "mnist_test.txt")
mnist_csv_to_custom_format("mnist_train.csv", "mnist_train.txt")
