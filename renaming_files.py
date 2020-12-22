'''
files have long calltype/recid/usvid/ch/.png file names
cleaning up these file names - retaining only usvid/ch/.png
'''

import os

os.chdir('./spectrograms')

for f in os.listdir():
    f_name, f_ext = os.path.splitext(f)
    try:
        calltype, recid, usvid, ch = f_name.split('_')
        new_name = '{}_{}{}'.format(usvid, ch, f_ext)
        os.rename(f, new_name)
    except:
        Exception
        pass
