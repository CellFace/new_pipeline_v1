# coding=utf-8

# Standard Library Imports
import os
import sys
import time
import timeit
import datetime
import json
import multiprocessing
import gc

# Third Party Imports
import numpy as np
import pandas as pd
import cv2
import h5py
import torch
from torchvision.utils import save_image

# Local Application/Library Specific Imports
from base.settings import LoggerMixin
from base.reconstructor import Reconstructor
from base.detection import ObjectDetection
from base.aggregate_analyzer import Aggregate
from base.utils import create_dir
import PySpin


class Acquisition(LoggerMixin):

    def __init__(self, app, config):
        super().__init__()
        self.holo_images = []
        self.ref_holo = None
        self.phase_images = []
        self.image_ids = []
        self.amp_images = []
        self.app = app
        self.continue_recording = True
        self.config = config
        self.rbc_count = 0
        self.wbc_count = 0
        self.plt_count = 0
        self.plt_plt_count = 0
        self.wbc_plt_count = 0
        self.wbc_wbc_count = 0
        self.aggregate_class_count = []

    def preview_images(self, cam: object):

        try:
            while self.continue_recording:
                try:
                    image_result = cam.GetNextImage(1000)

                    if not self.continue_recording:
                        break

                    if image_result.IsIncomplete():
                        self.logger.info(f'Image incomplete with image status {image_result.GetImageStatus()} ...')
                    else:                    
                        image_data = image_result.GetNDArray()
                        image_data = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        resized_image = cv2.resize(image_data, (1024, 768))
                        cv2.imshow('Image', resized_image)

                        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                            self.continue_recording = False
                            break

                        if cv2.waitKey(20) & 0xFF == ord('q'):
                            self.continue_recording = False
                            break

                    image_result.Release()

                except PySpin.SpinnakerException as ex:
                    self.logger.error(f'Error: {ex}')
                    return False

        except PySpin.SpinnakerException as ex:
            self.logger.error(f'Error: {ex}')
            return False

        cv2.destroyAllWindows()
        return True

    def get_images_slice(self, start_idx: int, end_idx: int):

        if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
            return ValueError('Invalid Data.')
        elif len(self.holo_images) == 0:
            return ValueError('There is no data.')
        elif end_idx > len(self.holo_images):
            end_idx = len(self.holo_images)
        return self.holo_images[start_idx:end_idx]

    def reconstruct_images(self, save_phase_images: bool, save_amp_images: bool, h5_file_path: str):
        """
            Reconstructs holographic images and optionally saves phase and amplitude images.

            Args:
                save_phase_images (bool): If True, saves reconstructed phase images.
                save_amp_images (bool): If True, saves reconstructed amplitude images.
                h5_file_path (str): Path to the H5 file containing holographic images.

            Process:
                - Initializes the Reconstructor with configured GPU and network paths.
                - Determines the number of batches for reconstruction based on batch size.
                - Raises an error if no images are available for reconstruction.
                - Normalizes holographic images and reconstructs them in batches.
                - Measures reconstruction speed and logs performance metrics.
                - Saves phase and amplitude images if specified.
        """
        reconstruct = Reconstructor(device = self.config["reconstruction_gpu"], fin_net_path = self.config["fin_net_path"], cnn_net_path = self.config["cnn_net_path"])
        self.logger.info(f'Total Images to reconstruct: {len(self.holo_images)}')
        batch_size = self.config["batch_size"]
        if len(self.holo_images) % batch_size == 0:
            num_batches = len(self.holo_images) // batch_size
        else:
            num_batches = (len(self.holo_images) // batch_size) + 1
        self.logger.info(f'Number of times to reconstruct: {num_batches}')
        # Check if num_batches is zero and raise an error if it is
        if num_batches == 0:
            raise ValueError("Number of Batch is zero, please ensure there are enough images to process.")

        phase_output_directory = os.path.join(self.config["img_output"], 'phase')
        amp_output_directory = os.path.join(self.config["img_output"], 'amp')
        os.makedirs(phase_output_directory, exist_ok=True)
        os.makedirs(amp_output_directory, exist_ok=True)

        self.holo_images = np.stack(self.holo_images)
        self.ref_holo = self.ref_holo[None, :].astype(float) / 255
        recon_speed = []
        for batch_index in range(num_batches):
            holo_batch = self.holo_images[batch_index * batch_size: (batch_index + 1) * batch_size]
            holo_batch = holo_batch.astype(float) / 255  # Normalize the batch outside the timing to focus on reconstruction time

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            phase_imgs, amp_imgs = reconstruct.reconstruct(holo_batch, ref_holo=self.ref_holo)
            phase_imgs *= 10
            amp_imgs *= 10
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
            recon_speed.append(elapsed_time / batch_size)                

            # Iterate over reconstructed images to save and collect
            for image_index, (phase_image_tensor, amp_image_tensor) in enumerate(zip(phase_imgs, amp_imgs)):
                phase_image = phase_image_tensor.detach().cpu().numpy()
                amp_image = amp_image_tensor.detach().cpu().numpy()
                self.phase_images.append(phase_image)  
                self.amp_images.append(amp_image)  

                # Calculate the correct ID based on batch_index and image_index
                image_id = batch_index * batch_size + image_index
                
                if save_phase_images:
                    phase_image = np.clip(phase_image, 0, 1)
                    phase_image_uint8 = (255 * phase_image).astype(np.uint8)
                    phase_image_uint8 = cv2.cvtColor(phase_image_uint8, cv2.COLOR_GRAY2BGR) if phase_image.ndim == 2 else phase_image_uint8
                    phase_output_file_path = os.path.join(phase_output_directory, f'phase_img_{image_id}.png')
                    cv2.imwrite(phase_output_file_path, phase_image_uint8)

                if save_amp_images:
                    amp_image = np.clip(amp_image, 0, 1)
                    amp_image_uint8 = (255 * amp_image).astype(np.uint8)
                    amp_image_uint8 = cv2.cvtColor(amp_image_uint8, cv2.COLOR_GRAY2BGR) if amp_image.ndim == 2 else amp_image_uint8
                    amp_output_file_path = os.path.join(amp_output_directory, f'amp_img_{image_id}.png')
                    cv2.imwrite(amp_output_file_path, amp_image_uint8)

        self.logger.info(f'Reconstruction speed: {recon_speed}')
        self.logger.info(f'Reconstruction speed per frame {np.mean(recon_speed):.3f} ms, std {np.std(recon_speed):.2f}')
                        
        self.holo_images = []

    def load_h5file(self, h5_file_path: str):
        """
            Loads holographic images from an H5 file.

            Args:
                h5_file_path (str): Path to the H5 file containing hologram data.

            Process:
                - Opens the H5 file in read mode.
                - Extracts and stores the reference hologram (`/calibrationImage/data`).
                - Iterates through time steps (`/timeStep{n}/hologram/data`) and loads hologram images.
                - Appends each hologram image to `self.holo_images` after reshaping.
        """
        time_step = 0
        with h5py.File(h5_file_path, 'r') as f:
            self.ref_holo = np.array(f['/calibrationImage/data']).reshape(1536, 2048)
            while True:
                path = f'/timeStep{time_step}/hologram/data'
                if path in f:
                    ds = f[path]
                    if isinstance(ds, h5py.Dataset):
                        self.holo_images.append(np.array(ds).reshape(1536, 2048))
                    time_step += 1
                else:
                    break

    def process_data(self, img_path, ovizio_reconstruction: bool, png_upload: bool):
        """
            Processes image data by performing object detection and aggregate analysis.

            Args:
                img_path (str): Path to the image or dataset to be processed.
                ovizio_reconstruction (bool): If True, uses the Ovizio detection model.
                png_upload (bool): If True, processes uploaded PNG images instead of reconstructed ones.

            Process:
                - Selects the appropriate detection model based on input parameters.
                - Performs object detection on the images using the selected model.
                - Logs detection time and initializes an `Aggregate` analyzer.
                - Identifies aggregates and logs processing time per frame.
                - Saves detection and aggregate analysis results in CSV and H5 files (if enabled).
                - Resets aggregate counts and clears image buffers after processing.
        """
        obj_detection = ObjectDetection()
        if ovizio_reconstruction or png_upload:
            weight_path = self.config["ovizio_detection_model"]
        else:
            weight_path = self.config["ai_detection_model"]
        self.logger.info(f'Measurement Directory: {img_path}')
        self.logger.info(f'Detection weight: {weight_path}')

        detect_start_time = time.perf_counter()
        phase_images, results, image_ids = obj_detection.predict(img_path, weight_path, reconstructed_images=self.phase_images, device=self.config["detection_gpu"])
        detect_end_time = time.perf_counter()

        self.log_detection_time(detect_start_time, detect_end_time, phase_images)

        analyzer_start_time = time.perf_counter()
        analyzer = Aggregate(phase_images, self.amp_images, img_path, results, image_ids)
        analyzer_end_time = time.perf_counter()
        analyzer_time_per_frame = 1000 * ((analyzer_end_time - analyzer_start_time) / len(phase_images))
        self.logger.info(f"The Initialize Analyzer Time: {analyzer_time_per_frame} ms per frame")
        # analyzer = Aggregate(phase_images, phase_images, img_path, results, image_ids)
        agg_start_time = time.perf_counter()
        aggregates_list, aggregate_image_ids, wbc_image_ids, aggregate_image_info = analyzer.find_aggregates(save_predicted_aggs=self.config["save_predicted_aggs"],
                                                                                            save_predicted_wbc=self.config["save_predicted_wbc"],
                                                                                            save_predicted_plt=self.config["save_predicted_plt"],
                                                                                            save_predicted_rbc=self.config["save_predicted_rbc"] )
        agg_end_time = time.perf_counter()
        agg_time_per_frame = 1000 * ((agg_end_time - agg_start_time) / len(phase_images))
        self.logger.info(f"The FindAggregate Time: {agg_time_per_frame} ms per frame")
        
        self.log_aggregate_info(aggregates_list)

        self.save_results_in_csv(img_path, phase_images, aggregate_image_ids, wbc_image_ids, aggregate_image_info)
        self.save_in_h5_if_enabled(phase_images, analyzer, img_path, aggregate_image_ids, aggregate_image_info)

        self.reset_aggregate_counts()
        self.phase_images = []
        self.amp_images = []
        phase_images = None
        results = None
        self.logger.info("Process completed!")

    def log_detection_time(self, start_time, end_time, images):
        batch_size = len(images) if images else 1
        time_per_frame = 1000 * (end_time - start_time) / batch_size
        self.logger.info(f"The Detection Time: {time_per_frame} milliseconds per frame")

    def log_aggregate_info(self, aggregates_list):
        
        for info in aggregates_list:
            self.rbc_count += info['counts_rbc']
            self.wbc_count += info['counts_wbc']
            self.plt_count += info['counts_plt']
            self.plt_plt_count += info['counts_plt_plt']
            self.wbc_plt_count += info['counts_wbc_plt']
            self.wbc_wbc_count += info['counts_wbc_wbc']
            self.aggregate_class_count += info['aggregate_class_counts']
        self.logger.info(f"RBC Count: {self.rbc_count}, WBC Count: {self.wbc_count}, PLT Count: {self.plt_count}, PLT-PLT Count: {self.plt_plt_count}, WBC-PLT Count: {self.wbc_plt_count}, WBC-WBC Count: {self.wbc_wbc_count}")

    def save_results_in_csv(self, img_path, images, aggregate_image_ids, wbc_image_ids, aggregate_image_info):
        results_data = {
            'Measurement Directory': [img_path],
            'Number of frames': [len(images)],
            'RBC Count': [self.rbc_count],
            'WBC Count': [self.wbc_count],
            'PLT Count': [self.plt_count],
            'PLT-PLT AGG Count': [self.plt_plt_count],
            'WBC-PLT AGG Count': [self.wbc_plt_count],
            'WBC-WBC AGG Count': [self.wbc_wbc_count],
            'Aggregate Image IDs': [aggregate_image_ids],
            'WBC Image IDs': [wbc_image_ids],
            'Aggregate Image Info': [aggregate_image_info],
        }
        # Create a DataFrame from the results data
        df = pd.DataFrame(results_data)
        
        # Ensure the output directory exists
        os.makedirs(self.config["img_output"], exist_ok=True)

        # Specify the path to the CSV output file
        csv_output_path = os.path.join(self.config["img_output"], 'results.csv')

        # Check if the file already exists
        file_exists = os.path.exists(csv_output_path)

        if file_exists:
            # Read the existing data
            existing_df = pd.read_csv(csv_output_path)
            # Append the new data to it
            df = pd.concat([existing_df, df], ignore_index=True)

        # Save the dataframe as a CSV file
        df.to_csv(csv_output_path, index=False)

    def save_in_h5_if_enabled(self, images, analyzer, img_path, aggregate_image_ids, aggregate_image_info):
        
        crop_image_ids, patch_ids, targeted_phase_images, targeted_amp_images, phase_patches, amp_patches, bboxes, class_label = analyzer.crop_process()
        bboxes = np.array(bboxes, dtype=np.int32)
        
        if self.config["containerization"]:
            self.logger.info("Create h5 file to store data.")
            time_now = datetime.datetime.now()
            formatted_time = time_now.strftime("%Y-%m-%d_%H-%M-%S")
            output_directory = str(self.config["img_output"])
            h5_file_name = f"{output_directory}/{formatted_time}.h5"
            with h5py.File(h5_file_name, 'w') as hf:
                hf.create_group('metadata')
                group2 = hf.create_group('Patch Data')
                group2.create_dataset('image_ids', data= crop_image_ids)
                group2.create_dataset('patch_ids', data=patch_ids,)
                group2.create_dataset('targeted_phase_images', data=targeted_phase_images)
                group2.create_dataset('targeted_amp_images', data=targeted_amp_images)
                group2.create_dataset('phase_patches', data=phase_patches)
                group2.create_dataset('amp_patches', data=amp_patches)
                group2.create_dataset('bboxes', data=bboxes)
                group2.create_dataset('labels', data=class_label)
                group3 = hf.create_group('Statistics')
                group3.create_dataset('Measurement Directory', data=img_path)
                group3.create_dataset('num_frames', data=len(images))
                group3.create_dataset('rbc_count', data=self.rbc_count)
                group3.create_dataset('wbc_count', data=self.wbc_count)  
                group3.create_dataset('plt_count', data=self.plt_count)
                group3.create_dataset('plt_plt_count', data=self.plt_plt_count)
                group3.create_dataset('wbc_plt_count', data=self.wbc_plt_count)
                group3.create_dataset('wbc_wbc_count', data=self.wbc_wbc_count)
                group3.create_dataset('aggregate_image_ids', data=aggregate_image_ids)
                aggregate_image_info_str = json.dumps(aggregate_image_info)
                group3.create_dataset('aggregate_image_info', data=np.string_(aggregate_image_info_str))

        crop_image_ids, patch_ids, targeted_phase_images, targeted_amp_images, phase_patches, amp_patches, bboxes, class_label = None, None, None, None, None, None, None, None
        self.logger.info("Data file creation completed!")

    def reset_aggregate_counts(self):
        
        for attr in ['rbc_count', 'wbc_count', 'plt_count', 'plt_plt_count', 'wbc_plt_count', 'wbc_wbc_count']:
            setattr(self, attr, 0)
        self.aggregate_class_count = []

    def reconstruction_worker(self, h5_file_path, phase_images, amp_images, condition, done_signal):
        """
            Performs batch-based holographic image reconstruction in a worker thread.

            Args:
                h5_file_path (str): Path to the H5 file containing holographic images.
                phase_images (list): Shared list to store reconstructed phase images.
                amp_images (list): Shared list to store reconstructed amplitude images.
                condition (threading.Condition): Synchronization object for notifying other threads.
                done_signal (multiprocessing.Value): Shared boolean flag indicating reconstruction completion.

            Process:
                - Loads holographic images from the H5 file.
                - Computes the number of batches based on the configured batch size.
                - Initializes the `Reconstructor` with the specified GPU and network paths.
                - Normalizes and reconstructs images in batches using the reference hologram.
                - Measures reconstruction speed and logs performance metrics.
                - Saves reconstructed images and updates shared lists (`phase_images`, `amp_images`).
                - Uses threading condition to notify other threads after processing each batch.
                - Sets the `done_signal` flag to True upon completion.
        """
        load_start_time = timeit.default_timer()
        self.load_h5file(h5_file_path)
        load_file_duration = timeit.default_timer() - load_start_time
        try:
            self.logger.info(f"Capture File Load Time: {load_file_duration:.2f} s")
        except Exception as e:
            print(e)
        self.logger.info(f'Total Images to reconstruct: {len(self.holo_images)}')

        batch_size = self.config["batch_size"]
        if len(self.holo_images) % batch_size == 0:
                num_batches = len(self.holo_images) // batch_size
        else:
            num_batches = (len(self.holo_images) // batch_size) + 1
        reconstructor = Reconstructor(device=self.config["reconstruction_gpu"], fin_net_path=self.config["fin_net_path"], cnn_net_path=self.config["cnn_net_path"])
        self.holo_images = np.stack(self.holo_images)
        self.ref_holo = self.ref_holo[None, :].astype(float) / 255
        
        recon_speed = []
        for batch_index in range(num_batches):
            self.logger.info(f"batch index: {batch_index}")
            holo_batch = self.holo_images[batch_index * batch_size: (batch_index + 1) * batch_size]
            holo_batch = holo_batch.astype(float) / 255  # Normalize batch

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            phase_imgs, amp_imgs = reconstructor.reconstruct(holo_batch, ref_holo=self.ref_holo)
            phase_imgs *= 10
            amp_imgs *= 10

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            recon_speed.append(elapsed_time / batch_size)    
            
            with condition:
                for image_index, (phase_img, amp_img) in enumerate(zip(phase_imgs, amp_imgs)):
                    phase_img = phase_img.detach().cpu().numpy()
                    amp_img = amp_img.detach().cpu().numpy()
                    phase_images.append(phase_img)
                    amp_images.append(amp_img)
                    image_id = batch_index * batch_size + image_index
                    self.save_images(phase_img, amp_img, image_id)
                condition.notify_all()
        self.logger.info(f"Total reconstructed phase images Line 454: {len(phase_images)}")
        self.logger.info(f'Reconstruction speed per frame {np.mean(recon_speed):.3f} ms, std {np.std(recon_speed):.2f}')

        with condition:
            done_signal.value = True
            condition.notify_all()

    def detection_worker(self, phase_images, amp_images, condition, file_path, done_signal):
        """
            Performs object detection on reconstructed images in a worker thread.

            Args:
                phase_images (list): Shared list containing phase images for detection.
                amp_images (list): Shared list containing amplitude images for detection.
                condition (threading.Condition): Synchronization object for coordinating thread execution.
                file_path (str): Path to the image dataset or directory.
                done_signal (multiprocessing.Value): Shared flag indicating when reconstruction is complete.

            Process:
                - Waits for reconstructed images to be available.
                - Retrieves a batch of images from shared lists (`phase_images`, `amp_images`).
                - Performs object detection using the selected AI model.
                - Logs detection time per frame and stores results.
                - Runs an aggregate analysis to identify detected objects.
                - Saves detection results in CSV and optionally in an H5 file.
                - Cleans up temporary data and resets aggregate counts.
        """
        obj_detection = ObjectDetection()
        detection_batch_size = self.config["detection_batch_size"]
        total_detect_time = []
        image_id_index = 0
        detected_phase_images, detected_amp_images, results, image_ids = [], [], [], []

        while True:
            with condition:
                while len(phase_images) < detection_batch_size and not done_signal.value:
                    condition.wait()
                if done_signal.value and len(phase_images) == 0:
                    break
                batch_size_to_process = min(detection_batch_size, len(phase_images))
                batch = [(phase_images.pop(0), amp_images.pop(0)) for _ in range(batch_size_to_process)]

            if not batch:
                continue

            phase_imgs, amp_imgs = zip(*batch)

            try:
                detect_start_time = time.time()
                phase_img_batch, result_batch, image_id_batch = obj_detection.predict(
                    img_path=file_path, 
                    reconstructed_images=phase_imgs, 
                    weight_path=self.config["ai_detection_model"], 
                    device=self.config["detection_gpu"]
                )
                image_id_batch = list(range(image_id_index * detection_batch_size, (image_id_index + 1) * detection_batch_size))
                detected_phase_images.extend(phase_img_batch)
                detected_amp_images.extend(amp_imgs)
                results.extend(result_batch)
                image_ids.extend(image_id_batch)
                image_id_index += 1
                detect_duration = time.time() - detect_start_time
                total_detect_time.append((detect_duration * 1000) / detection_batch_size)
            except Exception as e:
                self.logger.error(f'Error during detection: {e}')

            torch.cuda.empty_cache()
            gc.collect()

        self.logger.info(f'Total Detection Time per frame {np.mean(total_detect_time)} ms, std {np.std(total_detect_time)}')

        analyzer = Aggregate(detected_phase_images, detected_amp_images, file_path, results, image_ids)
        aggregates_list, aggregate_image_ids, wbc_image_ids, aggregate_image_info = analyzer.find_aggregates(
            save_predicted_aggs=self.config["save_predicted_aggs"],
            save_predicted_wbc=self.config["save_predicted_wbc"],
            save_predicted_plt=self.config["save_predicted_plt"],
            save_predicted_rbc=self.config["save_predicted_rbc"]
        )
        self.log_aggregate_info(aggregates_list)

        self.save_results_in_csv(file_path, detected_phase_images, aggregate_image_ids, wbc_image_ids, aggregate_image_info)
        self.save_in_h5_if_enabled(detected_phase_images, analyzer, file_path, aggregate_image_ids, aggregate_image_info)
        self.reset_aggregate_counts()

        detected_phase_images = []
        detected_amp_images = []
        image_ids = []
        results = None

        self.logger.info("Process completed!")

    def run_parallel_reconstruction_and_detection(self, file_path):
        """
            Runs holographic image reconstruction and object detection in parallel using multiprocessing.

            Args:
                file_path (str): Path to the dataset or H5 file containing holographic images.

            Process:
                - Initializes shared multiprocessing resources (`Manager`).
                - Creates shared lists (`phase_images`, `amp_images`) for inter-process communication.
                - Sets up a `Condition` object for thread synchronization.
                - Uses a shared `done_signal` flag to indicate when reconstruction is complete.
                - Spawns two parallel processes:
                    - `reconstruction_worker`: Loads and reconstructs holographic images.
                    - `detection_worker`: Detects objects in reconstructed images.
                - Starts both processes and waits for them to complete.
        """
        with multiprocessing.Manager() as manager:
            phase_images = manager.list()
            amp_images = manager.list()
            condition = manager.Condition()
            done_signal = manager.Value('b', False)  

            reconstruction_process = multiprocessing.Process(target=self.reconstruction_worker, args=(file_path, phase_images, amp_images, condition, done_signal))
            detection_process = multiprocessing.Process(target=self.detection_worker, args=(phase_images, amp_images, condition, file_path, done_signal))

            reconstruction_process.start()
            detection_process.start()

            reconstruction_process.join()
            detection_process.join()

    def save_images(self, phase_img, amp_img, image_id):

        phase_output_directory = os.path.join(self.config["img_output"], 'phase')
        amp_output_directory = os.path.join(self.config["img_output"], 'amp')
        os.makedirs(phase_output_directory, exist_ok=True)
        os.makedirs(amp_output_directory, exist_ok=True)

        if self.config["save_phase_images"]:
            phase_img = np.clip(phase_img, 0, 1)
            phase_img_uint8 = (255 * phase_img).astype(np.uint8)
            phase_img_uint8 = cv2.cvtColor(phase_img_uint8, cv2.COLOR_GRAY2BGR) if phase_img.ndim == 2 else phase_img_uint8
            phase_output_file_path = os.path.join(phase_output_directory, f'phase_img_{image_id}.png')
            try:
                cv2.imwrite(phase_output_file_path, phase_img_uint8)
            except Exception as e:
                self.logger.error(f"Failed to save phase image {image_id}: {e}")

        if self.config["save_amp_images"]:
            amp_img = np.clip(amp_img, 0, 1)
            amp_img_uint8 = (255 * amp_img).astype(np.uint8)
            amp_img_uint8 = cv2.cvtColor(amp_img_uint8, cv2.COLOR_GRAY2BGR) if amp_img.ndim == 2 else amp_img_uint8
            amp_output_file_path = os.path.join(amp_output_directory, f'amp_img_{image_id}.png')
            try:
                cv2.imwrite(amp_output_file_path, amp_img_uint8)
            except Exception as e:
                self.logger.error(f"Failed to save amplitude image {image_id}: {e}")