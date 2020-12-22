'''
images from each sex randomly assigned to training/validation/test subfolders
'''

import os
import shutil
import random

seed = 42
random.seed(seed)

training = 'training_data/'
validation = 'validation_data/'
test = 'test_data/'

# creating folders and subfolders
os.makedirs(training + 'female', exist_ok=True)
os.makedirs(training + 'male', exist_ok=True)

os.makedirs(validation + 'female', exist_ok=True)
os.makedirs(validation + 'male', exist_ok=True)

os.makedirs(test + 'female', exist_ok=True)
os.makedirs(test + 'male', exist_ok=True)

#to maintain count of specs/first initialize to 0
training_examples = validation_examples = test_examples = 0

for img in os.listdir('./female_unsorted'):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == '.png':
        random_num = random.random()
        if random_num < 0.8:
            location = './training/'
            training_examples += 1

        elif random_num < 0.9:
            location = './validation/'
            validation_examples += 1

        else:
            location = './test/'
            test_examples += 1

        shutil.copy('./female_unsorted/' + f_name + f_ext,
                    location + '/female/' + f_name + f_ext)


for img in os.listdir('./male_unsorted'):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == '.png':
        random_num = random.random()
        if random_num < 0.8:
            location = './training/'
            training_examples += 1

        elif random_num < 0.9:
            location = './validation/'
            validation_examples += 1

        else:
            location = './test/'
            test_examples += 1

        shutil.copy('./male_unsorted/' + f_name + f_ext,
                    location + '/male/' + f_name + f_ext)

print(f'Number of training examples: {training_examples}')
print(f'Number of validation examples: {validation_examples}')
print(f'Number of test examples: {test_examples}')
