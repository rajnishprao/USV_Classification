'''
file sorts spectrogram images by estrous state based on csv file
'''

import os
import shutil
import csv

# creating folders

os.makedirs('./proestrous', exist_ok=True)
os.makedirs('./estrous', exist_ok=True)
os.makedirs('./metestrous', exist_ok=True)
os.makedirs('./diestrous', exist_ok=True)

# spectrograms where caller = 'implanted'

for img in os.listdir('./female_unsorted'):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == '.png':
        estrous_info = csv.reader(open('estrous_implanted_calls.csv'))
        estrous_info.__next__()
        for line in estrous_info:
            usvid = line[0]
            if usvid == f_name[:6]:
                estrous_state = line[-1]
                if f_name.endswith('_ch1') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch1') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch1') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch1') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)

print('Implanted spectograms sorted')

# spectrograms where caller = 'stimulus'

for img in os.listdir('./female_unsorted'):
    f_name, f_ext = os.path.splitext(img)
    if f_ext == '.png':
        estrous_info = csv.reader(open('estrous_stimulus_calls.csv'))
        estrous_info.__next__()
        for line in estrous_info:
            usvid = line[0]
            if usvid == f_name[:6]:
                estrous_state = line[-1]
                if f_name.endswith('_ch1') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'pr':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './proestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch1') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'es':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './estrous/' + f_name + f_ext)
                elif f_name.endswith('_ch1') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'me':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './metestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch1') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch2') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch3') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)
                elif f_name.endswith('_ch4') and estrous_state == 'di':
                    shutil.copy('./female_unsorted/' + f_name + f_ext, './diestrous/' + f_name + f_ext)

print('Stimulus spectograms sorted')
