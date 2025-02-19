# coding=utf-8

# Local Application/Library Specific Imports
from base.utils import load_config, setup_logger
import PySpin


class LoggerMixin:

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(
            name="my_logger",
            log_dir=self.config["log_output"],
            log_filename=self.config["log_file"],
            max_size=self.config["log_file_size"],
            backup_count=self.config["backup_count"]
        )


class CameraParameters(LoggerMixin):

    def __init__(self):
            super().__init__()

    def set_acquisition_mode(self, nodemap, mode='Continuous'):
        """
            Sets the camera's acquisition mode using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure settings.
                mode (str, optional): The acquisition mode to set. Valid options are:
                    - 'SingleFrame'
                    - 'MultiFrame'
                    - 'Continuous' (default)

            Process:
                - Validates the provided acquisition mode.
                - Retrieves the acquisition mode node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the corresponding mode entry and verifies its availability.
                - Sets the acquisition mode to the specified value.
                - Logs the updated acquisition mode.
        """
        # check if the mode is valid
        if mode not in ('SingleFrame', 'MultiFrame', 'Continuous'):
            raise ValueError("Invalid mode. Please provide a valid mode ('SingleFrame', 'MultiFrame', 'Continuous')")
    
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            raise PySpin.SpinnakerException(f'Unable to set acquisition mode to {mode} (enum retrieval). Aborting...')

        # Retrieve entry node from enumeration node
        entry_acquisition_mode = node_acquisition_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_acquisition_mode) or not PySpin.IsReadable(entry_acquisition_mode):
            raise PySpin.SpinnakerException(f'Unable to set acquisition mode to {mode} (entry retrieval). Aborting...')

        # Retrieve integer value from entry node
        acquisition_mode = entry_acquisition_mode.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode)

        self.logger.info(f'Acquisition mode set to  {mode}')

    def set_pixel_format(self, nodemap, format='Mono8'):
        """
            Sets the camera's pixel format using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure settings.
                format (str, optional): The pixel format to set. Default is 'Mono8'.

            Process:
                - Retrieves the `PixelFormat` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the desired pixel format entry from the enumeration node.
                - Verifies if the format entry is available and readable.
                - Retrieves the integer value corresponding to the pixel format.
                - Sets the new pixel format for the camera.
                - Logs the updated pixel format.
        """
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if not PySpin.IsAvailable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
            raise PySpin.SpinnakerException(f'Unable to set the PixelFormat to {format} (enum retrieval). Aborting...')
        
        # Retrieve the desired entry node from the enumeration node
        entry_pixel_format = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(format))
        if not PySpin.IsAvailable(entry_pixel_format) or not PySpin.IsReadable(entry_pixel_format):
            raise PySpin.SpinnakerException(f'Unable to set the PixelFormat to {format} (entry retrieval). Aborting...')
        # Retrieve the integer value from the entry node
        pixel_format = entry_pixel_format.GetValue()

        # Set integer as new value for enumeration node
        node_pixel_format.SetIntValue(pixel_format)
        self.logger.info(f'PixelFormat set to {format}')

    def set_bufferhandling_mode(self, sNodemap, mode='NewestOnly'):
        """
            Sets the camera's stream buffer handling mode using the Spinnaker SDK.

            Args:
                sNodemap (PySpin.INodeMap): The node map of the camera's stream settings.
                mode (str, optional): The buffer handling mode to set. Default is 'NewestOnly'.
                    Valid options depend on the camera's supported buffer handling modes.

            Process:
                - Retrieves the `StreamBufferHandlingMode` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the specified buffer handling mode from the enumeration node.
                - Verifies if the mode entry is available and readable.
                - Retrieves the corresponding integer value for the buffer handling mode.
                - Sets the new buffer handling mode for the camera.
                - Logs the updated buffer handling mode.
        """
        # Set bufferhandling mode
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            raise PySpin.SpinnakerException(f'Unable to set stream buffer handling mode to {mode} (enum retrieval). Aborting...')

        # Retrieve entry node from enumeration node
        entry_buffer_mode = node_bufferhandling_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_buffer_mode) or not PySpin.IsReadable(entry_buffer_mode):
            raise PySpin.SpinnakerException(f'Unable to set stream buffer handling mode to {mode} (entry retrieval). Aborting...')

        # Retrieve integer value from entry node
        get_node_mode = entry_buffer_mode.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(get_node_mode)
        self.logger.info(f'Buffer Handling Mode is set to {mode} {get_node_mode}')

    def set_auto_exposure_mode(self, nodemap, mode='Off'):
        """
            Sets the camera's auto-exposure mode using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure exposure settings.
                mode (str, optional): The auto-exposure mode to set. Default is 'Off'.
                    Valid options:
                    - 'Off': Disables auto-exposure.
                    - 'Once': Enables auto-exposure for a single adjustment.
                    - 'Continuous': Enables continuous auto-exposure adjustment.

            Process:
                - Validates the provided mode.
                - Retrieves the `ExposureAuto` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the corresponding entry for the selected mode.
                - Verifies if the entry is available and readable.
                - Retrieves the integer value of the mode and sets it.
                - Logs the new auto-exposure mode.

            Returns:
                bool: True if the mode is successfully set, False if an invalid mode is provided.
        """
        if mode not in ['Off', 'Once', 'Continuous']:
            self.logger.info(f'Invalid mode {mode} selected. Please choose one of the following modes: Off, Once, Continuous')
            return False
    
        # Set Auto Exposure mode
        node_exposure_mode = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(node_exposure_mode) or not PySpin.IsWritable(node_exposure_mode):
            raise PySpin.SpinnakerException(f'Unable to set Exposure Auto to {mode} (enum retrieval). Aborting...')

        entry_exposure_mode = node_exposure_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_exposure_mode) or not PySpin.IsReadable(entry_exposure_mode):
            raise PySpin.SpinnakerException(f'Unable to set Exposure Auto to {mode} (entry retrieval). Aborting...')

        exposure_auto_mode = entry_exposure_mode.GetValue()

        node_exposure_mode.SetIntValue(exposure_auto_mode)

        self.logger.info(f'Auto Exposure mode set to {mode} ')
        return True

    def set_exposure(self, nodemap, exp_seconds):
        """
            Sets the camera's exposure time manually using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure exposure settings.
                exp_seconds (float): Desired exposure time in microseconds.

            Returns:
                bool: True if the exposure time is successfully set, False if the requested value is out of range.

            Process:
                - Disables auto-exposure mode by calling `set_auto_exposure_mode(nodemap, mode='Off')`.
                - Retrieves the minimum and maximum supported exposure times from the camera.
                - Validates the requested exposure time against the supported range.
                - Retrieves the `ExposureTime` node from the nodemap.
                - Checks if the node is available and writable.
                - Sets the exposure time to the specified value.
                - Logs the updated exposure time.
        """
        self.set_auto_exposure_mode(nodemap, mode='Off')
        
        # Get minimum and maximum exposure time values supported by the camera
        node_exp_time_min = PySpin.CFloatPtr(nodemap.GetNode('ExposureTimeAbs'))
        min_exp_time = node_exp_time_min.GetMin()
        node_exp_time_max = PySpin.CFloatPtr(nodemap.GetNode('ExposureTimeAbs'))
        max_exp_time = node_exp_time_max.GetMax()
        
        if exp_seconds < min_exp_time or exp_seconds > max_exp_time:
            self.logger.info(f'Exposure time of {exp_seconds} microseconds is not within the supported range of the camera. The supported range is between {min_exp_time} and {max_exp_time} microseconds.')
            return False

        # Set Exposure Time to less than 1/50th of a second (5000 micro second is used as an example)
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            raise PySpin.SpinnakerException(f'\nUnable to set Exposure Time (float retrieval). Aborting...\n')

        node_exposure_time.SetValue(exp_seconds)
        self.logger.info(f'Exposure Set to {exp_seconds}')
        return True

    def set_auto_gain(self, nodemap, mode='Off'):
        """
            Sets the camera's auto-gain mode using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure gain settings.
                mode (str, optional): The auto-gain mode to set. Default is 'Off'.
                    Valid options:
                    - 'Off': Disables auto-gain.
                    - 'Once': Enables auto-gain for a single adjustment.
                    - 'Continuous': Enables continuous auto-gain adjustment.

            Returns:
                bool: True if the mode is successfully set, False if an invalid mode is provided.

            Process:
                - Validates the provided mode.
                - Retrieves the `GainAuto` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the corresponding entry for the selected mode.
                - Verifies if the entry is available and readable.
                - Retrieves the integer value of the mode and sets it.
                - Logs the new auto-gain mode.
        """
        if mode not in ['Off', 'Once', 'Continuous']:
            self.logger.info(f'Invalid mode {mode} selected. Please choose one of the following modes: Off, Once, Continuous')
            return False
        
        node_gain_mode = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsAvailable(node_gain_mode) or not PySpin.IsWritable(node_gain_mode):
            raise PySpin.SpinnakerException(f'Unable to set Gain Auto to {mode} (enum retrieval). Aborting...')

        entry_gain_mode = node_gain_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_gain_mode) or not PySpin.IsReadable(entry_gain_mode):
            raise PySpin.SpinnakerException(f'Unable to set Gain Auto to {mode} (entry retrieval). Aborting...')

        gain_auto_mode = entry_gain_mode.GetValue()

        node_gain_mode.SetIntValue(gain_auto_mode)

        self.logger.info(f'Auto Gain mode set to {mode} ')
        return True

    def set_gain(self, nodemap, gain_val: int):
        """
            Sets the camera's gain manually using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure gain settings.
                gain_val (int): Desired gain value to be set.

            Returns:
                bool: True if the gain is successfully set, False otherwise.

            Process:
                - Retrieves the minimum and maximum supported gain values from the camera.
                - Validates the requested gain value against the supported range.
                - Disables auto-gain mode by calling `set_auto_gain(nodemap, mode='Off')`.
                - Retrieves the `Gain` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the gain value to the specified amount.
                - Logs the updated gain setting.
        """
        # Get minimum and maximum gain values supported by the camera
        node_gain_min = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        min_gain = node_gain_min.GetMin()
        node_gain_max = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        max_gain = node_gain_max.GetMax()
        
        if gain_val < min_gain or gain_val > max_gain:
            raise PySpin.SpinnakerException(f'Gain value of {gain_val} is not within the supported range of the camera. The supported range is between {min_gain} and {max_gain}.')

        # Turn off auto Gain
        self.set_auto_gain(nodemap, mode='Off')

        # Set Gain
        node_gain_val = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsAvailable(node_gain_val) or not PySpin.IsWritable(node_gain_val):
            raise PySpin.SpinnakerException('\nUnable to set Gain (float retrieval). Aborting...\n')

        node_gain_val.SetValue(gain_val)
        self.logger.info(f'Gain Value is set to {gain_val}')
        return True

    def enable_framerate(self, nodemap, enable: bool):
        """
            Enables or disables the camera's frame rate control using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure frame rate settings.
                enable (bool): If True, enables frame rate control; if False, disables it.

            Process:
                - Retrieves the `AcquisitionFrameRateEnable` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the frame rate enable value to the specified `enable` parameter.
        """
        enable_fps = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if not PySpin.IsAvailable(enable_fps) or not PySpin.IsWritable(enable_fps):
            raise PySpin.SpinnakerException('Unable to change the value for framerate. The AcquisitionFrameRateEnable node is not available or not writable.')
        enable_fps.SetValue(enable)

    def acquisition_framerate(self, nodemap):
        """
            Retrieves the camera's current acquisition frame rate using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to access frame rate settings.

            Returns:
                float: The current acquisition frame rate.

            Process:
                - Retrieves the `AcquisitionFrameRate` node from the camera's nodemap.
                - Checks if the node is available and readable.
                - Returns the current frame rate value.
        """
        node_acquisition_fps = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_acquisition_fps) or not PySpin.IsReadable(node_acquisition_fps):
            raise PySpin.SpinnakerException('Unable to Get Acquisition Frame Rate. The AcquisitionFrameRate node is not available or not readable.')
        return node_acquisition_fps.GetValue()

    def get_resulting_framerate(self, nodemap):
        """
            Retrieves the camera's resulting frame rate using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to access frame rate settings.

            Returns:
                float or None: The resulting acquisition frame rate if available, otherwise None.

            Process:
                - Attempts to retrieve the `AcquisitionResultingFrameRate` node from the camera's nodemap.
                - Checks if the node is available and readable.
                - If available, retrieves and logs the frame rate.
                - If unavailable, logs an error and returns None.
                - Catches any Spinnaker-related exceptions and logs the error.
        """
        try:
            # Check if the AcquisitionResultingFrameRate node is available
            resulting_framerate_node = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionResultingFrameRate'))

            if PySpin.IsAvailable(resulting_framerate_node) and PySpin.IsReadable(resulting_framerate_node):
                resulting_framerate = resulting_framerate_node.GetValue()
                self.logger.info(f"Resulting frame rate: {resulting_framerate} fps.")
                return resulting_framerate
            else:
                self.logger.error("Unable to get resulting frame rate. The AcquisitionResultingFrameRate node is not available or not readable.")
                return None
        except PySpin.SpinnakerException as ex:
            self.logger.error(f"Error: {ex}")
            return None

    def set_gamma(self, nodemap, gamma_val: float):
        """
            Sets the camera's gamma correction value using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure gamma settings.
                gamma_val (float): Desired gamma value to be set. Must be less than 3.9.

            Returns:
                float: The newly set gamma value.

            Process:
                - Validates that the gamma value is within the allowed range (< 3.9).
                - Retrieves the `Gamma` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the gamma value to the specified amount.
                - Logs the updated gamma setting.
                - Returns the new gamma value for verification.
        """
        if  not (gamma_val < 3.9):
            raise ValueError(f"Gamma Value shoud be less than 3.9")
        node_gamma = PySpin.CFloatPtr(nodemap.GetNode('Gamma'))
        if not PySpin.IsAvailable(node_gamma) or not PySpin.IsWritable(node_gamma):
            raise PySpin.SpinnakerException('Unable to set Gamma value. The Gamma node is not available or not writable.')
        node_gamma.SetValue(gamma_val)
        self.logger.info(f'Gamma Value is set to {gamma_val}')
        return node_gamma.GetValue()

    def auto_sharp(self, nodemap, enable: bool):
        """
            Enables or disables the camera's automatic sharpening feature using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to access sharpening settings.
                enable (bool): If True, enables auto sharpening; if False, disables it.

            Returns:
                bool: The new sharpening auto state (True if enabled, False if disabled).

            Process:
                - Retrieves the `SharpeningAuto` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the sharpening auto state based on the `enable` parameter.
                - Returns the updated state of the sharpening feature.
        """
        node_sharp = PySpin.CBooleanPtr(nodemap.GetNode('SharpeningAuto'))
        if not PySpin.IsAvailable(node_sharp) or not PySpin.IsWritable(node_sharp):
            raise PySpin.SpinnakerException('Unable to change the value for auto sharpening feature. The SharpeningAuto node is not available or not writable.')
        node_sharp.SetValue(enable)
        return node_sharp.GetValue()

    def enable_sharpening(self, nodemap, enable: bool):
        """
            Enables or disables the camera's sharpening feature using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to access sharpening settings.
                enable (bool): If True, enables sharpening; if False, disables it.

            Returns:
                bool: The new sharpening state (True if enabled, False if disabled).

            Process:
                - Retrieves the `SharpeningEnable` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the sharpening state based on the `enable` parameter.
                - Returns the updated state of the sharpening feature.
        """
        node_sharp_enable = PySpin.CBooleanPtr(nodemap.GetNode('SharpeningEnable'))
        if not PySpin.IsAvailable(node_sharp_enable) or not PySpin.IsWritable(node_sharp_enable):
            raise PySpin.SpinnakerException('Unable to change the value for sharpening feature. The SharpeningEnable node is not available or not writable.')
        node_sharp_enable.SetValue(enable)
        return node_sharp_enable.GetValue()

    def set_sharpening(self, nodemap, sharp_val: float):
        """
            Sets the camera's sharpening level using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to access sharpening settings.
                sharp_val (float): Desired sharpening value. Must be between 1 and 8.

            Returns:
                float: The newly set sharpening value.

            Process:
                - Validates that `sharp_val` is within the allowed range (1 to 8).
                - Retrieves the `Sharpening` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the sharpening value to the specified amount.
                - Returns the updated sharpening value.
        """
        if not (1 <= sharp_val <= 8):
            raise ValueError("sharp_val should be between 1 to 8")
        node_sharp_val = PySpin.CFloatPtr(nodemap.GetNode('Sharpening'))
        if not PySpin.IsAvailable(node_sharp_val) or not PySpin.IsWritable(node_sharp_val):
            raise PySpin.SpinnakerException('Unable to set Sharpening. The Sharpening node is not available or not writable.')
        node_sharp_val.SetValue(sharp_val)
        return node_sharp_val.GetValue()

    def auto_white_balance(self, nodemap, mode='Off'):
        """
            Sets the camera's auto white balance mode using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure white balance settings.
                mode (str, optional): The auto white balance mode to set. Default is 'Off'.
                    Valid options:
                    - 'Off': Disables auto white balance.
                    - 'Once': Performs auto white balance adjustment once.
                    - 'Continuous': Continuously adjusts white balance.

            Process:
                - Validates that the provided mode is one of the allowed options.
                - Retrieves the `BalanceWhiteAuto` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the corresponding entry for the selected mode.
                - Verifies if the entry is available and readable.
                - Retrieves the integer value of the mode and sets it.
                - Logs the new auto white balance mode.
        """
        if mode not in ['Off', 'Once', 'Continuous']:
            raise ValueError("mode should be one of the options: 'Off', 'Once', 'Continuous'")
        node_whitebalance_mode = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        
        if not PySpin.IsAvailable(node_whitebalance_mode) or not PySpin.IsWritable(node_whitebalance_mode):
            raise PySpin.SpinnakerException(f'Unable to set BalanceWhiteAuto to {mode} (enumeration retrieval).')

        entry_whitebalance_mode = node_whitebalance_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_whitebalance_mode) or not PySpin.IsReadable(entry_whitebalance_mode):
            raise PySpin.SpinnakerException(f'Unable to set BalanceWhiteAuto to {mode} (entry retrieval).')

        white_balance_mode = entry_whitebalance_mode.GetValue()
        node_whitebalance_mode.SetIntValue(white_balance_mode)
        self.logger.info(f'Balance White Auto mode set to {mode}')  

    def auto_blacklevel(self, nodemap, mode='Off'):
        """
            Sets the camera's auto black level mode using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure black level settings.
                mode (str, optional): The auto black level mode to set. Default is 'Off'.
                    Valid options:
                    - 'Off': Disables auto black level adjustment.
                    - 'Once': Adjusts black level once.
                    - 'Continuous': Continuously adjusts black level.

            Returns:
                bool: True if the mode is successfully set, False otherwise.

            Process:
                - Retrieves the `BlackLevelAuto` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the corresponding entry for the selected mode.
                - Verifies if the entry is available and readable.
                - Retrieves the integer value of the mode and sets it.
                - Logs the new auto black level mode.
        """
        node_autoblack_mode = PySpin.CEnumerationPtr(nodemap.GetNode('BlackLevelAuto'))
        if not PySpin.IsAvailable(node_autoblack_mode) or not PySpin.IsWritable(node_autoblack_mode):
            self.logger.warning('\nUnable to set Auto Blacklevel mode. Aborting...\n')
            return False

        entry_autoblack_mode = node_autoblack_mode.GetEntryByName(mode)
        if not PySpin.IsAvailable(entry_autoblack_mode) or not PySpin.IsReadable(entry_autoblack_mode):
            self.logger.warning('\nUnable to set Auto Blacklevel Mode. Aborting...\n')
            return False

        autoblack_mode = entry_autoblack_mode.GetValue()

        node_autoblack_mode.SetIntValue(autoblack_mode)
        self.logger.info(f'AutoBlack is set to {mode} mode')

    def set_blacklevel(self, nodemap, black_val: float):
        """
            Sets the camera's black level value using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure black level settings.
                black_val (float): The desired black level value to set.

            Process:
                - Retrieves the `BlackLevel` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Sets the black level to the specified value.
                - Logs the updated black level setting.
        """
        node_black_level = PySpin.CFloatPtr(nodemap.GetNode('BlackLevel'))
        if not PySpin.IsAvailable(node_black_level) or not PySpin.IsWritable(node_black_level):
            raise PySpin.SpinnakerException('Unable to set BlackLevel. The BlackLevel node is not available or not writable.')

        node_black_level.SetValue(black_val)
        self.logger.info(f'BlackLevel Set to {black_val}')


class DeviceInfo(LoggerMixin):

    def __init__(self):
            super().__init__()

    def get_device_info(self, nodemap):
        """
            Retrieves and logs the camera's device information using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to access device information.

            Returns:
                bool: True if the device information was successfully retrieved, False if an error occurred.

            Process:
                - Attempts to retrieve the `DeviceInformation` category node from the camera's nodemap.
                - Checks if the node is available and readable.
                - Iterates through all device features within the category.
                - Checks if each feature is readable and logs its name and value.
                - If a feature is not readable, logs a corresponding message.
                - Catches any Spinnaker-related exceptions and logs the error.
        """
        try:
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    if PySpin.IsReadable(node_feature):
                        feature_name = node_feature.GetName()
                        feature_value = node_feature.ToString()
                        self.logger.info(f'{feature_name}: {feature_value}')
                    else:
                        self.logger.info(f'{node_feature.GetName()} is not readable')

        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False

        return True


class BufferHandlingControl(LoggerMixin):

    def __init__(self):
        super().__init__()

    def buffer_count_mode(self, nodemap, mode='Manual'):
        """
            Sets the camera's stream buffer count mode using the Spinnaker SDK.

            Args:
                nodemap (PySpin.INodeMap): The node map of the camera to configure buffer settings.
                mode (str, optional): The buffer count mode to set. Default is 'Manual'.
                    Valid options depend on the camera's supported buffer count modes.

            Returns:
                bool: True if the buffer count mode is successfully set, False otherwise.

            Process:
                - Retrieves the `StreamBufferCountMode` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Retrieves the entry corresponding to the selected mode.
                - Verifies if the entry is available and readable.
                - Retrieves the integer value of the mode and sets it.
                - Logs the new buffer count mode.
                - Catches and logs any Spinnaker-related exceptions.
        """
        try:
            # Retrieve the node for the StreamBufferCountMode
            node_buffer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferCountMode'))
            
            # Check if the node is available and writable
            if not PySpin.IsAvailable(node_buffer_mode) or not PySpin.IsWritable(node_buffer_mode):
                raise PySpin.SpinnakerException(f'Unable to set Stream Buffer Count Mode to {mode} (enum retrieval). Aborting...')

            # Retrieve the entry for the desired buffer count mode
            entry_buffer_mode = node_buffer_mode.GetEntryByName(mode)
            
            # Check if the entry is available and readable
            if not PySpin.IsAvailable(entry_buffer_mode) or not PySpin.IsReadable(entry_buffer_mode):
                raise PySpin.SpinnakerException(f'Unable to set Stream Buffer Count Mode to {mode} (entry retrieval). Aborting...')

            # Get the value of the buffer count mode
            buffer_count_mode = entry_buffer_mode.GetValue()

            # Set the buffer count mode
            node_buffer_mode.SetIntValue(buffer_count_mode)

            print(f'Stream Buffer Count mode set to {mode} ')
            return True
        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False

    def set_buffer_count(self, sNodemap, num_buffer: int):
        """
            Sets the manual buffer count for the camera's stream using the Spinnaker SDK.

            Args:
                sNodemap (PySpin.INodeMap): The node map of the camera to configure buffer settings.
                num_buffer (int): The desired number of buffers to allocate.

            Returns:
                bool: True if the buffer count is successfully set, False if an error occurs.

            Process:
                - Retrieves the `StreamBufferCountManual` node from the camera's nodemap.
                - Checks if the node is available and writable.
                - Logs the default and maximum buffer count values.
                - Sets the buffer count to the specified `num_buffer` value.
                - Logs the updated buffer count.
                - Catches and logs any Spinnaker-related exceptions.
        """
        try:
            # Retrieve the node for the StreamBufferCountManual
            buffer_count = PySpin.CIntegerPtr(sNodemap.GetNode('StreamBufferCountManual'))
            
            # Check if the node is available and writable
            if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
                raise PySpin.SpinnakerException('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')

            self.logger.info(f'Default Buffer Count: {buffer_count.GetValue()}')
            self.logger.info(f'Maximum Buffer Count: {buffer_count.GetMax()}')

            # Set the buffer count to the desired value
            buffer_count.SetValue(num_buffer)
            self.logger.info(f'Buffer count now set to: {buffer_count.GetValue()}')
            return True
        
        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False

