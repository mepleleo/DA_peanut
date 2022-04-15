# DA_peanut

## paper: Identification of Moldy Peanuts under Different Varieties and Moisture Content Using Hyperspectral Imaging and Data Augmentation Technologies

### classification model:   
KNN/SVM/MobileViT-xs

### data augmentation methods:

1. pixel classifier(KNN/SVM)   
Erasing   
Noise   
TSW   
DSM(proposed method)   

2. image classifier(MobileViT-xs)   
Erasing   
Noise   
Rotation   
DSM(proposed method)   

## Method principle
generated spectra = original spectra + λ × spectral difference   
generated hyperspectral image = rotate(original image + λ × spectral difference)   
λ is a parameter used to adjust the offset of the original data.   
For spectral difference:   
The data of this class was divided into two parts with the median as the boundary, and the average spectrum of each part was calculated for generating the spectral difference of two parts.

