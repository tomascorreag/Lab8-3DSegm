# Lab8 - 3D Segmentation
This lab is an extension of what we learned 2 weeks ago, but using a different kind of data. We will use a method we designed for semantic segmentation of 3D medical images, especifically for the tasks of brain tumor (multimodal MRI), kidney and kidney tumor (CT) segmentation, and we will apply it to the task of left atrium segmentation in single-modality MRI.

## Brain Tumor Segmentation
<p align="center"><img src="https://sites.google.com/site/braintumorsegmentation/_/rsrc/1431350972218/home/brats_data.png" width="600"/></p>

Gliomas are the most common primary brain malignancies, with different degrees of aggressiveness, variable prognosis and various heterogeneous histological sub-regions, i.e. peritumoral edema, necrotic core, enhancing and non-enhancing tumor core. BraTS utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Due to this highly heterogeneous appearance and shape, segmentation of brain tumors in multimodal MRI scans is one of the most challenging tasks in medical image analysis. [1]

### Kidney and Kidney Tumor Segmentation
<p align="center"><img src="https://wiki.cancerimagingarchive.net/download/attachments/61081171/c4kc-kits.png" width="400"/></p>

There are more than 400,000 new cases of kidney cancer each year [2], and surgery is its most common treatment [3]. Due to the wide variety in kidney and kidney tumor morphology, there is currently great interest in how tumor morphology relates to surgical outcomes, [4,5] as well as in developing advanced surgical planning techniques [6]. Automatic semantic segmentation is a promising tool for these efforts, but morphological heterogeneity makes it a difficult problem.

### Left Atrium Segmentation

<p align="center"><img src="https://www.cardiacatlas.org/wp-content/uploads/2015/11/la_segmentation_figure.png" width="400"/></p>


The left atrium is clinically important for the management of atrial fibrillation in patients. Segmentation can be used to generate anatomical models that can be employed in guided treatment and also more recently for cardiac biophysical modelling. [7]

## This lab
You will run the codes I gave you in that new dataset and use those results as baseline. After that, you will perform **any experiment you want** (changes in the architecture or training schedule and parameters) to see how much you can improve. The important thing here is that I want a complete (at least 6 results beside the baseline) and logical experimental framework. In other words, I want to know why you did what you did and why you thought it would work. Finally, I want you to validate the best method you obtain using 4 fold cross-validation.

Dataset:
You don't have to download it, the Data is already stored in the machine (if we can bring Malik back to life soon, I will store the data in that folder)
```
Malik: /media/sda1/vision2020_01/Data
BCV001: /media/user_home2/vision2020_01/Data
```

### Libraries
To read the data:
```conda install -c conda-forge nibabel```
Apex: 
https://github.com/NVIDIA/apex

If you want to do data augmentation I recommend this library:
https://github.com/MIC-DKFZ/batchgenerators


## References
1. B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d
2. “Kidney Cancer Statistics.” World Cancer Research Fund, 12 Sept. 2018, www.wcrf.org/dietandcancer/cancer-trends/kidney-cancer-statistics.
3. “Cancer Diagnosis and Treatment Statistics.” Stages | Mesothelioma | Cancer Research UK, 26 Oct. 2017, www.cancerresearchuk.org/health-professional/cancer-statistics/diagnosis-and-treatment.
4. Kutikov, Alexander, and Robert G. Uzzo. "The RENAL nephrometry score: a comprehensive standardized system for quantitating renal tumor size, location and depth." The Journal of urology 182.3 (2009): 844-853.
5. Ficarra, Vincenzo, et al. "Preoperative aspects and dimensions used for an anatomical (PADUA) classification of renal tumours in patients who are candidates for nephron-sparing surgery." European urology 56.5 (2009): 786-793.
6. Taha, Ahmed, et al. "Kid-Net: Convolution Networks for Kidney Vessels Segmentation from CT-Volumes." arXiv preprint arXiv:1806.06769 (2018).
7. Tobon-Gomez C, Geers AJ, Peters, J, Weese J, Pinto K, Karim R, Ammar M, Daoudi A, Margeta J, Sandoval Z, Stender B, Zheng Y, Zuluaga, MA, Betancur J, Ayache N, Chikh MA, Dillenseger J-L, Kelm BM, Mahmoudi S, Ourselin S, Schlaefer A, Schaeffter T, Razavi R, Rhode KS. Benchmark for Algorithms Segmenting the Left Atrium From 3D CT and MRI Datasets. IEEE Transactions on Medical Imaging, 34(7):1460–1473, 2015.

## Deadline
April 29, 11:59pm
