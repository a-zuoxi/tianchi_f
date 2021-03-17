import pandas
import os
import shutil
import tqdm

jir=pandas.read_csv(r'../data/imagenet_round1_210122/dev.csv')

save_folder = '../data/imagenet/clean/'
org_folder = '../data/imagenet_round1_210122/images/'

for i in tqdm.tqdm(range(len(jir['ImageId']))):
    to_folder = os.path.join(save_folder, str(jir['TrueLabel'][i]))
    org_file = os.path.join(org_folder, jir['ImageId'][i])
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    to_file = os.path.join(to_folder, jir['ImageId'][i])
    shutil.copyfile(org_file, to_file)
