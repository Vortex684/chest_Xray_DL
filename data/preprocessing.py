import pandas as pd 
import numpy as np 

BASE = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'

ALL_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


BBOX_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax'
]

BBOX_DISEASE_IDX = {disease: idx for idx, disease in enumerate(BBOX_DISEASES)}


def load_bbox_data(base_path):
    bbox_df=pd.read_csv(f'{base_path}/BBox_List_2017.csv')
    # Group all boxes by image filename
    # One image can have multiple rows — multiple diseases
    
    bbox_lookup={}
    
    for _,row in bbox_df.iterrows():
        f_name = row['Image Index']
        disease = row['Finding Label']
        
        if disease not in BBOX_DISEASE_IDX:
            continue 
        
        if f_name not in bbox_lookup:
            bbox_lookup[f_name]=[]
            
        bbox_lookup[f_name].append({
            'disease':disease,
            'x': row['Bbox [x'],
            'y': row['y'],
            'w': row['w'],
            'h': row['h]']
        })   
        
    return bbox_lookup
            
def create_mask(f_name, bbox_lookup, img_size=224, original_size=1024):
    
    mask=np.zeros((img_size,img_size,len(BBOX_DISEASES)),dtype=np.float32)
    
    if f_name not in bbox_lookup:
        return mask
    
    scale = img_size / original_size            
    
    for box in bbox_lookup[f_name]:
        
        channel=BBOX_DISEASE_IDX[box['disease']]
                
        x  = int(box['x'] * scale)
        y  = int(box['y'] * scale)
        w  = int(box['w'] * scale)
        h  = int(box['h'] * scale)
                
        x2 = min(x + w, img_size)
        y2 = min(y + h, img_size)
        
        mask[y:y2, x:x2, channel] = 1.0
        
    return mask
    
def encode_labels(finding_labels_str):

    diseases = finding_labels_str.split('|')
    return [1 if d in diseases else 0 for d in ALL_DISEASES ]   