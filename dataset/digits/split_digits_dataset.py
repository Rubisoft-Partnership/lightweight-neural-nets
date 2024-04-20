from sklearn.model_selection import train_test_split

# Read the data from digits.txt
with open("digits.txt", "r") as f:
    data = f.readlines()

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Write the training data to train.txt
with open("train.txt", "w") as f:
    f.writelines(train_data)

# Write the testing data to test.txt
with open("test.txt", "w") as f:
    f.writelines(test_data)

print("Done")