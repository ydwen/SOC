## **Baseline for Deep Face Recognition**

This package gives an example for deep face recognition, including data, code, and trained models. Specifically, we train the deep model on CAISA dataset and test on MegaFace. More details please refers to 

    Wen Y, Zhang K, Li Z, et al. A discriminative feature learning approach for deep face recognition[C]//European Conference on Computer Vision. Springer International Publishing, 2016: 499-515.



### **Files**
You can view the folder tree using a simple command.
```Shell
cd $THIS_PACKAGE
# It is assumed that all the commands is executed in the root directory of this package
 
tree -L 3
```

The folder tree should be like this.
> - Code
    - Test
        - MegaFace
    - Train
        - caffe-face
        - models
- Data
    - Test
        - MegaFace
    - Train
        - CAISAdataset_112X96_#1
        - CAISAdataset_112X96_#2
        - CAISAdataset_112X96_#3
        - +List
- Result
    - Test
        - 28-Net
    - Train
        - 28-Net
### **Training**
1 -- Install caffe-face. The Installation completely the same as [Caffe](http://caffe.berkeleyvision.org/). Please follow the [installation instructions](http://caffe.berkeleyvision.org/installation.html). Make sure you have correctly installed before using our code. 

``` Shell
  # In your Makefile.config, make sure to specify the directory of Matlab
  MATLAB_DIR := (path_to_your_matlab)
  # It is recommended to use CUDNN
  USE_CUDNN := 1
```
```Shell
  make all -j16 && make matcaffe
```

2 -- Train model by CAISA

```Matlab
 # In Code/Train/models/28-Net/train.prototxt, specify the patch for training. (default: CAISA, patch 2)
 Line 15: source: "Data/Train/+List/caisa_train_#2.txt"
 # In Code/Train/models/28-Net/solver.prototxt, specify the directory for storing models. (default: Result/Train/28-Net/)
```
``` Shell
 ./Code/Train/caffe-face/build/tools/caffe train -solver Code/Train/models/28-Net/solver.prototxt -gpu x,y
```
 
### **Testing**

1 -- Create binary files for MegaFace (as required by the provided development kit)
```Matlab
    # In Code/Test/MegaFace/test_megaface.m, specify the patch for testing, project name, model, weights.
   
   run test_megaface.m in Matlab
```
   You will get `facescrub_patch_2.mat`, `flickr_patch_2.mat`, `fgnet_patch_2.mat`, storing in `Result/Test/28-Net/`
   
```Matlab
    # In Code/Test/MegaFace/creat_binary.m, specify the directory for binary files, project name, selected features. 
    
    run creat_binary.m in Matlab
```
2 -- Test on MegaFace
```Shell
    cd Code/Test/MegaFace/devkit/experiment/ 
    
    ./test_facescrub 28-Net 28-Net 1000000
```
