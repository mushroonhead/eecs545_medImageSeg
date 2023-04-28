Requirements:
- Please see "requirements.txt"

General Instructions:
- Instructions on data processing:
    -> place Training Nifti data in "data/task_data/TestVolumes_full" and "data/task_data/TestLabels_full"
    -> place Training Nifti data in "data/task_data/TrainVolumes_full" and "data/task_data/TrainLabels_full"
- Running:
    -> Run UNet, ResUNet, SegResNet in ONE-HOT encoding mode (1 label per voxel): $ python train_one_hot.py
    -> Run UNet, ResUNet, SegResNet in multi-class label mode (tumorous liver has tumor and liver tag): $ python train_two_class.py
    -> Run UNETR: see UNETR_+_MONAI_LiTS_segmentation_3d.ipynb