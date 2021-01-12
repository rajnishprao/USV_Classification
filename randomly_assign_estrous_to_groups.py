"""
images from each estrous state randomly assigned to training/validation/test subfolders
"""

import os
import random
import shutil

seed = 42
random.seed(seed)

training = "training_data/"
validation = "validation_data/"
test = "test_data/"

# creating folders and subfolders
os.makedirs(training + "pr", exist_ok=True)
os.makedirs(training + "es", exist_ok=True)
os.makedirs(training + "me", exist_ok=True)
os.makedirs(training + "di", exist_ok=True)

os.makedirs(validation + "pr", exist_ok=True)
os.makedirs(validation + "es", exist_ok=True)
os.makedirs(validation + "me", exist_ok=True)
os.makedirs(validation + "di", exist_ok=True)

os.makedirs(test + "pr", exist_ok=True)
os.makedirs(test + "es", exist_ok=True)
os.makedirs(test + "me", exist_ok=True)
os.makedirs(test + "di", exist_ok=True)


# to maintain count of specs/first initialize to 0

training_examples = validation_examples = test_examples = 0

for img in os.listdir("./proestrous"):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == ".png":
        random_num = random.random()
        if random_num < 0.70:
            location = "./training_data/"
            training_examples += 1

        elif random_num < 0.85:
            location = "./validation_data/"
            validation_examples += 1

        else:
            location = "./test_data/"
            test_examples += 1

        shutil.copy(
            "./proestrous/" + f_name + f_ext, location + "/pr/" + f_name + f_ext
        )


for img in os.listdir("./estrous"):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == ".png":
        random_num = random.random()
        if random_num < 0.70:
            location = "./training_data/"
            training_examples += 1

        elif random_num < 0.85:
            location = "./validation_data/"
            validation_examples += 1

        else:
            location = "./test_data/"
            test_examples += 1

        shutil.copy("./estrous/" + f_name + f_ext, location + "/es/" + f_name + f_ext)


for img in os.listdir("./metestrous"):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == ".png":
        random_num = random.random()
        if random_num < 0.70:
            location = "./training_data/"
            training_examples += 1

        elif random_num < 0.85:
            location = "./validation_data/"
            validation_examples += 1

        else:
            location = "./test_data/"
            test_examples += 1

        shutil.copy(
            "./metestrous/" + f_name + f_ext, location + "/me/" + f_name + f_ext
        )


for img in os.listdir("./diestrous"):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == ".png":
        random_num = random.random()
        if random_num < 0.70:
            location = "./training_data/"
            training_examples += 1

        elif random_num < 0.85:
            location = "./validation_data/"
            validation_examples += 1

        else:
            location = "./test_data/"
            test_examples += 1

        shutil.copy("./diestrous/" + f_name + f_ext, location + "/di/" + f_name + f_ext)

print(f"Number of training examples: {training_examples}")
print(f"Number of validation examples: {validation_examples}")
print(f"Number of test examples: {test_examples}")
