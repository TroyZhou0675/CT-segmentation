import os
import random
from collections import defaultdict
from skimage.io import imread

def get_patient_id_from_filename(filename):
    return filename.split('_')[0]


def get_partitions(images_dir='./new_data/images/', masks_dir = './new_data/masks/',split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}):
    all_filenames = os.listdir(images_dir)
    patients_id = sorted(set(get_patient_id_from_filename(image) for image in all_filenames))

    random.seed(42)
    random.shuffle(patients_id)

    num_patients = len(patients_id)
    train_end = int(split_ratios['train'] * num_patients)
    val_end = train_end + int(split_ratios['val'] * num_patients)
    
    train_ids = patients_id[:train_end]
    val_ids = patients_id[train_end:val_end]
    test_ids = patients_id[val_end:]



    partition_images = defaultdict(list)
    partition_masks = defaultdict(list)

    train_findings = set(train_ids)
    val_findings = set(val_ids)
    test_findings = set(test_ids)

    for filename in all_filenames:
        patient_id = get_patient_id_from_filename(filename)
        if patient_id in train_findings:
            partition_images['train'].append(images_dir+filename)
        elif patient_id in val_findings:
            partition_images['val'].append(images_dir+filename)
        elif patient_id in test_findings:
            partition_images['test'].append(images_dir+filename)


    all_masknames = os.listdir(masks_dir)  # 替换为您的掩码目录路径
    for maskname in all_masknames:
        patient_id = get_patient_id_from_filename(maskname)
        if patient_id in train_findings:
            partition_masks['train'].append(masks_dir+maskname)
        elif patient_id in val_findings:
            partition_masks['val'].append(masks_dir+maskname)
        elif patient_id in test_findings:
            partition_masks['test'].append(masks_dir+maskname)

    return partition_images, partition_masks