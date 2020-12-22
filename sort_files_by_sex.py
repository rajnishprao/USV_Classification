'''
file sorts spectrogram images by sex based on csv file
'''

import os
import shutil
import csv

# creating folders

os.makedirs('./female_unsorted', exist_ok=True)
os.makedirs('./male_unsorted', exist_ok=True)


# spectrograms where caller_sex = 'female'

for img in os.listdir('./spectrograms'):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == '.png':
        sex_info = csv.reader(open('calls_modif.csv'))
        sex_info.__next__()
        for line in sex_info:
            usvid = line[3]
            if usvid == f_name[:6]:
                caller_sex = line[-1]
                if f_name.endswith('_ch1') and caller_sex == 'female':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './female_unsorted/' + f_name + f_ext)

                elif f_name.endswith('_ch2') and caller_sex == 'female':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './female_unsorted/' + f_name + f_ext)

                elif f_name.endswith('_ch3') and caller_sex == 'female':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './female_unsorted/' + f_name + f_ext)

                elif f_name.endswith('_ch4') and caller_sex == 'female':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './female_unsorted/' + f_name + f_ext)

print('Spectrograms from females sorted')

# spectrograms where caller_sex = 'male'

for img in os.listdir('./spectrograms'):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == '.png':
        sex_info = csv.reader(open('calls_modif.csv'))
        sex_info.__next__()
        for line in sex_info:
            usvid = line[3]
            if usvid == f_name[:6]:
                caller_sex = line[-1]
                if f_name.endswith('_ch1') and caller_sex == 'male':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './male_unsorted/' + f_name + f_ext)

                elif f_name.endswith('_ch2') and caller_sex == 'male':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './male_unsorted/' + f_name + f_ext)

                elif f_name.endswith('_ch3') and caller_sex == 'male':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './male_unsorted/' + f_name + f_ext)

                elif f_name.endswith('_ch4') and caller_sex == 'male':
                    shutil.copy('./spectrograms/' + f_name + f_ext, './male_unsorted/' + f_name + f_ext)

print('Spectrograms from males sorted')
