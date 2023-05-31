# BlobDetection
 A tool for Hannah to count mayfly eggs. Reuires a running and up-to-date version of Anaconda and Jupyter notebooks on your local PC. (I will try to get it running on Binder from here when I have a chance)
 
You can download this repository as a folder to your machine. One you have unzipped the array, navigate to the unpacked directory using your terminal. In the directory, run the command:
 conda env create -n ENVNAME --file BlobDetection.yml
^ this command will create a working environmnent in your anaconda and will download all the necessary python packages for the code to work. After the environment BlobDetection is installed, you will need to run the following code in your terminal before using the code:
  conda activate BlobDetection
^ this command activates the working environment
  jupyter notebook
^ this command will launch a jupyter notebook interface in your browser tab. In that tab, you should be able to navigate to the directory and run the file count_eggs.ipynb. Everything will be done through that file and it contains some superficial instructions and tuning controls for you to play with. Note that you do not need to move all your photos to the directory, it should be possible to upload images from any directory of your machine.
Currently, the code is adjusted for the image in the input_photos folder called testimage.jpg. If you run all cells in order without making any changes to the detector tuning parameters, you should get 82 eggs in that image. (in reality there is more but that is why you have control over the tuning parameters).
 
