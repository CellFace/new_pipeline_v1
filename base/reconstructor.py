# coding=utf-8

# Third Party Imports
import torch
from einops import rearrange
import torchvision.transforms as transform


class Reconstructor:
    
    def __init__(self, device, fin_net_path, cnn_net_path, input_size=(1536, 2048), output_size=(384, 512)):
        self.device = device
        self.traced_fin = torch.jit.load(fin_net_path).to(self.device)
        self.traced_cnn = torch.jit.load(cnn_net_path).to(self.device)
        self.ip_size = input_size
        self.op_size = output_size
        self.downsize = transform.Resize(size=self.op_size)

    def reconstruct(self, holo, ref_holo=None):
        """
            Performs hologram reconstruction using a neural network model.

            Args:
                holo (np.ndarray): Input hologram image(s) in a 3D NumPy array (batch, height, width).
                ref_holo (np.ndarray, optional): Reference hologram for phase correction.
                    - If None, the mean of the input hologram is used as the reference.

            Process:
                - Converts the input hologram to a PyTorch tensor and ensures its values are scaled between 0 and 1.
                - Validates input dimensions to match the expected size.
                - If a reference hologram is provided, it is processed and validated; otherwise, the mean of the input hologram is used.
                - Stacks the reference hologram with the input hologram.
                - Applies a Fourier Transform (FFT) to the hologram and processes it using a pre-trained neural network.
                - Uses a CNN to refine the prediction and extract phase and amplitude information.
                - Synchronizes CUDA operations for GPU-based computations.

            Returns:
                tuple:
                    - `pred[:, 0, ...]` (torch.Tensor): Reconstructed phase images.
                    - `pred[:, 1, ...]` (torch.Tensor): Reconstructed amplitude images.
        """
        holo = torch.from_numpy(holo)
        assert (
            torch.max(holo) <= 1
        ), "Input hologram range should be 0~1. Please rescale it."
        assert len(holo.shape) == 3, "Input is not a 3D array."
        assert holo.size(1) == self.ip_size[0], "Unexpected input size."
        assert holo.size(2) == self.ip_size[1], "Unexpected input size."

        if ref_holo is not None:
            ref_holo = torch.from_numpy(ref_holo)
            assert torch.max(ref_holo) <= 1, "Reference hologram range should be 0~1."
            if len(ref_holo.shape) == 2:
                ref_holo = ref_holo[None]
            elif len(ref_holo.shape) == 3:
                # print('Multiple reference holograms detected. Taking the average.')
                ref_holo = torch.mean(ref_holo, dim=0, keepdim=True)
            else:
                print(f"Unexpected reference hologram shape: {ref_holo.shape}")
            assert (
                ref_holo.size(1) == self.ip_size[0]
            ), "Unexpected reference hologram size."
            assert (
                ref_holo.size(2) == self.ip_size[1]
            ), "Unexpected reference hologram size."
        else:
            print(
                "No reference hologram provided. Using the average of input hologram instead."
            )
            ref_holo = torch.mean(holo, dim=0, keepdim=True)

        holo = torch.cat((ref_holo, holo), dim=0).to(device=self.device, dtype=torch.float)
        with torch.no_grad():
            holo_resize = self.downsize(holo)
            holo_fft = torch.view_as_real(torch.fft.fft2(holo))

            holo_resize = rearrange(holo_resize, "b h w -> b 1 h w")
            holo_fft = rearrange(holo_fft, "b h w cx -> b 1 h w cx", cx=2)

            x = self.traced_fin(holo_fft)
            x = x[1:, ...] / x[:1, ...]
            x = torch.cat([x.angle(), x.abs()], dim=1)
            holo_resize = holo_resize[1:, ...]
            pred = self.traced_cnn(holo_resize, x)

        torch.cuda.synchronize()

        return pred[:, 0, ...], pred[:, 1, ...]
