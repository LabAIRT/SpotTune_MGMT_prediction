# SpotTune GBM Project
code used for the study of GBM related to the use of SpotTune: transfer learning with adaptive fine tuning

The code uses the [UPENN-GBM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642) dataset stored in the Cancer Imaging Archives

This code requires the MedicalNet repository as a dependency (for the weights)
The MedicalNet repository should be placed in the same directory as `gbm_project`


> `git clone https://github.com/Tencent/MedicalNet.git`


package requirements are stored in `enviornment.yml`

Notebooks are provided that go through the data prep, training, and evaluation process.
For proper running of the notebooks, they should be placed in the same directory as `gbm_project`

> `cd gbm_project`
>
> `cp ./notebooks/* ../`

The notebooks have dataset paths that should be replaced with your own local paths to the UPENN-GBM dataset
