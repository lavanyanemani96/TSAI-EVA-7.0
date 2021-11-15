import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

def augmentation(data, mu, sigma):

  if data == 'Train':
    transform = A.Compose([A.HorizontalFlip(),
                           A.ShiftScaleRotate(),
                           A.CoarseDropout(max_holes=1, 
                                           max_height=16, 
                                           max_width=16, 
                                           min_holes=1, 
                                           min_height=16,
                                           min_width=16,
                                           fill_value=np.mean(mu)),
                           A.ToGray(),
                           A.Normalize(mean=mu, std=sigma), 
                           ToTensorV2()])
  else:
    transform = A.Compose([A.Normalize(mean=mu, std=sigma), 
                           ToTensorV2()])
  
  return transform
