### Workflow
   +	*Image Capture:* Initially, users capture images utilizing the Spinnaker Software Development Kit (SDK). 
      This SDK provides a comprehensive interface for camera operation and image acquisition.
   +	*Image Reconstruction:* Following capture, these images undergo a reconstruction process using an AI-based reconstructor.
      This step involves sophisticated algorithms to enhance image quality and prepare the data for further analysis.
   +	*Object Detection:* The reconstructed images are then processed by an object detection model. 
      This model is designed to identify and classify specific objects within the images, using advanced machine learning techniques.
   +	*Data Storage:* Finally, all processed results are compiled and stored in a containerized file format. 
      This format ensures efficient organization and accessibility of the output data for subsequent use or analysis. 


### Installation
   + **Note:** Use python3.8, and create New Python environemnt with it.
   + ** Install spinnaker_python-2.7.0.128-cp38-cp38-win_amd64.whl file**
      Run the following command to install PySpin to your associated Python version.
      This command assumes you have your PATH variable set correctly for Python:
      ```python -m pip install spinnaker_python-2.7.0.128-cp38-cp38-win_amd64.whl```
      Ensure that the wheel downloaded matches the Python version you are installing to!
   + ** Install requirement.txt file**
      ```pip install -r requirements.txt```
   + ** Install ultralytics (YOLO)**
      Run this command to install ultralytics
      ```pip install ultralytics```
   + ** Install PyTorch**
      Please visit [the official PyTorch website](https://pytorch.org/) for installation instructions. Ensure you select the installation command that matches your operating system, package manager (like pip or conda), and whether you need GPU support. For example, to install PyTorch with CPU support using pip, you might use the command `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`. After installation, verify it by running `python -c "import torch; print(torch.__version__)"` in your terminal or command prompt, which should display the installed PyTorch version.
      Note: Make sure all of the installed packages (torch, torchaudio, torchvision) versions are compictable with your graphic card. 


### Usage:
   + **	Starting the Application:**
      - Sart Application: Run main_view.py file from "dhm/views" folder. If the installation is successful, the main GUI will pop up.

   + **	Configuration Setup Instructions:**
      - *Configuration Setup:* To do the configuration for the application, click “Configuration” button.
      - *Form Pop-Up:* Upon clicking, “Configuration Form” will appear.
      - *Define parameters:* Input values for parameters such as 'pixel_format', 'exposure_seconds', etc
      - *Specify Paths:*  - Within the configuration form, you must provide the paths to essential components for object detection and reconstruction. These include:
                        - The path to the object detection weight file.
                        - The paths to the reconstruction models.
                        - The path to the desired output folder for storing the results.
                        Providing accurate paths is essential for the proper functioning of the application's object detection and reconstruction features.
      - *Saving Parameters:* Click “Save Configuration” after defining the values for the parameters.

   + **	Analyzing h5 Capture Files:**
      - *Initiating Analysis:* To begin analyzing an h5 capture file (originating from OsOne), users should first click the “Process Capture File” button.
      - *Form Pop-Up:* Upon clicking, the “Process H5 Form” will appear.
      - *File Selection:* Users must then click “Select File” to browse and choose the desired h5 file from their PC’s drive.
      - *Reconstruction Method Selection:* Users have the option to select the reconstruction method—either Ovizio or AI.
      - *Processing the File:* After making the selections, click “Process File” to start the analysis of the data.
      - *Output Generation:* If the process completes successfully, the results will be generated as an h5 container file.

   + **	Analyzing Reconstructed PNG Images:**
      - *Providing Image Path:* Users should provide the folder path containing the reconstructed PNG phase images.
      - *Initiating Analysis:* Click “Process File” to begin the analysis of the data.
      - *Output Generation:* Similar to the h5 capture file process, if the analysis is successful, the results will be generated as an h5 container file.

   + **	Previewing Images:**
      - *Realtime Preview:* To see the real-time hologram images from the microscope, click “Preview Images”. Users can check the images by previewing them.


      - Configure Paths:
         - Update VALID_H5_PATH with the path where the H5 capture file is stored.
         - Update INPUT_PATH with the path where the ovizio reconstructed phase images are located.
         - For object detection, update VALID_IMG_PATH and VALID_WEIGHT_PATH with the 

   
