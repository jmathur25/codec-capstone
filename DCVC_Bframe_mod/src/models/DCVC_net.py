import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..utils.stream_helper import get_downsampled_shape
from ..layers.layers import MaskedConv2d, subpel_conv3x3


class DCVC_net(nn.Module):
    def __init__(self, up_strategy):
        assert up_strategy in ["resize_conv", "mod_convtranspose2d", "default"]
        super().__init__()
        out_channel_mv = 128
        out_channel_N = 64
        out_channel_M = 96

        self.out_channel_mv = out_channel_mv
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_z_mv = BitEstimator(out_channel_N)

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_encoder = GaussianEncoder()

        # self.mvEncoder = nn.Sequential(
        #     nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
        #     GDN(out_channel_mv),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        #     GDN(out_channel_mv),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        #     GDN(out_channel_mv),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        # )
        self.mvEncoder = nn.Sequential(
            nn.Conv2d(4, out_channel_mv * 2, 3, stride=2, padding=1),
            GDN(out_channel_mv * 2),
            nn.Conv2d(out_channel_mv * 2, out_channel_mv * 2, 3, stride=2, padding=1),
            GDN(out_channel_mv * 2),
            nn.Conv2d(out_channel_mv * 2, out_channel_mv * 2, 3, stride=2, padding=1),
            GDN(out_channel_mv * 2),
            nn.Conv2d(out_channel_mv * 2, out_channel_mv * 2, 3, stride=2, padding=1),
        )

        if up_strategy == "resize_conv":
            # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
            self.mvDecoder_part1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=0),
                GDN(out_channel_mv, inverse=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=0),
                GDN(out_channel_mv, inverse=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=0),
                GDN(out_channel_mv, inverse=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=0),
            )
        elif up_strategy == "mod_convtranspose2d":
            self.mvDecoder_part1 = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channel_mv,
                    out_channel_mv,
                    4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                GDN(out_channel_mv, inverse=True),
                nn.ConvTranspose2d(
                    out_channel_mv,
                    out_channel_mv,
                    4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                GDN(out_channel_mv, inverse=True),
                nn.ConvTranspose2d(
                    out_channel_mv,
                    out_channel_mv,
                    4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                GDN(out_channel_mv, inverse=True),
                nn.ConvTranspose2d(
                    out_channel_mv, 2, 4, stride=2, padding=1, output_padding=0
                ),
            )
        elif up_strategy == "default":
            # self.mvDecoder_part1 = nn.Sequential(
            #     nn.ConvTranspose2d(
            #         out_channel_mv,
            #         out_channel_mv,
            #         3,
            #         stride=2,
            #         padding=1,
            #         output_padding=1,
            #     ),
            #     GDN(out_channel_mv, inverse=True),
            #     nn.ConvTranspose2d(
            #         out_channel_mv,
            #         out_channel_mv,
            #         3,
            #         stride=2,
            #         padding=1,
            #         output_padding=1,
            #     ),
            #     GDN(out_channel_mv, inverse=True),
            #     nn.ConvTranspose2d(
            #         out_channel_mv,
            #         out_channel_mv,
            #         3,
            #         stride=2,
            #         padding=1,
            #         output_padding=1,
            #     ),
            #     GDN(out_channel_mv, inverse=True),
            #     nn.ConvTranspose2d(
            #         out_channel_mv, 2, 3, stride=2, padding=1, output_padding=1
            #     ),
            # )
            self.mvDecoder_part1 = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channel_mv * 2,
                    out_channel_mv * 2,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                GDN(out_channel_mv * 2, inverse=True),
                nn.ConvTranspose2d(
                    out_channel_mv * 2,
                    out_channel_mv * 2,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                GDN(out_channel_mv * 2, inverse=True),
                nn.ConvTranspose2d(
                    out_channel_mv * 2,
                    out_channel_mv * 2,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                GDN(out_channel_mv * 2, inverse=True),
                nn.ConvTranspose2d(
                    out_channel_mv * 2, 4, 3, stride=2, padding=1, output_padding=1
                ),
            )
        else:
            raise ValueError("Unhandled up strategy", up_strategy)

        self.mvDecoder_part2 = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )

        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N * 2 + 3, out_channel_N * 2, 5, stride=2, padding=2),
            GDN(out_channel_N * 2),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N * 2),
            nn.Conv2d(out_channel_N * 2, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N * 3, out_channel_N * 2, 3, stride=1, padding=1),
            ResBlock(out_channel_N * 2, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        if up_strategy == "resize_conv":
            # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
            self.priorDecoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_N, out_channel_M, 3, stride=1, padding=0),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_M, out_channel_M, 3, stride=1, padding=0),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_M, out_channel_M, 3, stride=1, padding=0),
            )
        elif up_strategy == "mod_convtranspose2d":
            self.priorDecoder = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channel_N,
                    out_channel_M,
                    4,
                    stride=2,
                    padding=2,
                    output_padding=2,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_M,
                    out_channel_M,
                    4,
                    stride=2,
                    padding=2,
                    output_padding=2,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_M, out_channel_M, 3, stride=1, padding=1
                ),
            )
        elif up_strategy == "default":
            self.priorDecoder = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channel_N,
                    out_channel_M,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_M,
                    out_channel_M,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_M, out_channel_M, 3, stride=1, padding=1
                ),
            )
        else:
            raise ValueError("Unhanded up strategy", up_strategy)

        # self.mvpriorEncoder = nn.Sequential(
        #     nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        # )
        self.mvpriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_mv * 2, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        if up_strategy == "resize_conv":
            # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
            self.mvpriorDecoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=0),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    out_channel_N, out_channel_N * 3 // 2, 3, stride=1, padding=0
                ),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    out_channel_N * 3 // 2, out_channel_mv * 2, 3, stride=1, padding=0
                ),
            )
        elif up_strategy == "mod_convtranspose2d":
            self.mvpriorDecoder = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channel_N,
                    out_channel_N,
                    4,
                    stride=2,
                    padding=2,
                    output_padding=2,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_N,
                    out_channel_N * 3 // 2,
                    4,
                    stride=2,
                    padding=2,
                    output_padding=2,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_N * 3 // 2, out_channel_mv * 2, 3, stride=1, padding=1
                ),
            )
        elif up_strategy == "default":
            # self.mvpriorDecoder = nn.Sequential(
            #     nn.ConvTranspose2d(
            #         out_channel_N,
            #         out_channel_N,
            #         5,
            #         stride=2,
            #         padding=2,
            #         output_padding=1,
            #     ),
            #     nn.LeakyReLU(inplace=True),
            #     nn.ConvTranspose2d(
            #         out_channel_N,
            #         out_channel_N * 3 // 2,
            #         5,
            #         stride=2,
            #         padding=2,
            #         output_padding=1,
            #     ),
            #     nn.LeakyReLU(inplace=True),
            #     nn.ConvTranspose2d(
            #         out_channel_N * 3 // 2, out_channel_mv * 2, 3, stride=1, padding=1
            #     ),
            # )
            self.mvpriorDecoder = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channel_N,
                    out_channel_N,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_N,
                    out_channel_N * 3,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channel_N * 3, out_channel_mv * 4, 3, stride=1, padding=1
                ),
            )
        else:
            raise ValueError("Unknown up strategy", up_strategy)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 15 // 3, out_channel_M * 12 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 9 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 9 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
            out_channel_M, out_channel_M * 2, kernel_size=5, padding=2, stride=1
        )

        self.auto_regressive_mv = MaskedConv2d(
            out_channel_mv * 2, out_channel_mv * 4, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 24 // 3, out_channel_mv * 20 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 20 // 3, out_channel_mv * 16 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 16 // 3, out_channel_mv * 12 // 3, 1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N * 2, 5, stride=2, padding=2),
            GDN(out_channel_N * 2),
            nn.Conv2d(out_channel_N * 2, out_channel_N * 2, 5, stride=2, padding=2),
            GDN(out_channel_N * 2),
            nn.Conv2d(out_channel_N * 2, out_channel_N * 2, 5, stride=2, padding=2),
            GDN(out_channel_N * 2),
            nn.Conv2d(out_channel_N * 2, out_channel_M * 2, 5, stride=2, padding=2),
        )

        self.opticFlow = ME_Spynet()

    def motioncompensation(self, ref, mv):
        ref_feature = self.feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context = self.context_refine(prediction_init)

        return context

    def mv_refine(self, ref, mv):
        return self.mvDecoder_part2(torch.cat((mv, ref), 1)) + mv

    def train_quantize(self, x):
        # Simulates quantization by adding uniform noise [-0.5, 0.5]
        return x + torch.rand(x.shape).to(x.device) - 0.5

    def quantize(self, inputs, mode, means, compress_type):
        assert mode == "dequantize"
        if compress_type == "no_compress":
            return inputs
        elif compress_type == "train_compress":
            return self.train_quantize(inputs)
        elif compress_type == "full":
            outputs = inputs.clone()
            outputs -= means
            outputs = torch.round(outputs)
            outputs += means
            return outputs
        else:
            raise ValueError("Unknown compress type", compress_type)

    def feature_probs_based_sigma(self, feature, mean, sigma, compress_type):
        outputs = self.quantize(feature, "dequantize", mean, compress_type)
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(
            torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)
        )
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(
            torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)
        )
        return total_bits, prob

    def iclr18_estrate_bits_z_mv(self, z_mv):
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - self.bitEstimator_z_mv(z_mv - 0.5)
        total_bits = torch.sum(
            torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)
        )
        return total_bits, prob

    def update(self, force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    def encode_decode(self, ref_frame, input_image, output_path):
        encoded = self.encode(ref_frame, input_image, output_path)
        decoded = self.decode(ref_frame, output_path)
        encoded["recon_image"] = decoded
        return encoded

    def encode(self, ref_frame, input_image, output_path):
        from ..utils.stream_helper import encode_p

        N, C, H, W = ref_frame.size()
        compressed = self.compress(ref_frame, input_image)
        mv_y_string = compressed["mv_y_string"]
        mv_z_string = compressed["mv_z_string"]
        y_string = compressed["y_string"]
        z_string = compressed["z_string"]
        encode_p(H, W, mv_y_string, mv_z_string, y_string, z_string, output_path)
        return {
            "bpp_mv_y": compressed["bpp_mv_y"],
            "bpp_mv_z": compressed["bpp_mv_z"],
            "bpp_y": compressed["bpp_y"],
            "bpp_z": compressed["bpp_z"],
            "bpp": compressed["bpp"],
        }

    def decode(self, ref_frame, input_path):
        from ..utils.stream_helper import decode_p

        height, width, mv_y_string, mv_z_string, y_string, z_string = decode_p(
            input_path
        )
        return self.decompress(
            ref_frame, mv_y_string, mv_z_string, y_string, z_string, height, width
        )

    def compress_ar(
        self, y, kernel_size, context_prediction, params, entropy_parameters
    ):
        kernel_size = 5
        padding = (kernel_size - 1) // 2

        height = y.size(2)
        width = y.size(3)

        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_q = torch.zeros_like(y)
        y_scales = torch.zeros_like(y)

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[0:1, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    context_prediction.weight,
                    bias=context_prediction.bias,
                )

                p = params[0:1, :, h : h + 1, w : w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                y_crop = y_crop[0:1, :, padding : padding + 1, padding : padding + 1]
                y_crop_q = torch.round(y_crop - means_hat)
                y_hat[0, :, h + padding, w + padding] = (y_crop_q + means_hat)[
                    0, :, 0, 0
                ]
                y_q[0, :, h, w] = y_crop_q[0, :, 0, 0]
                y_scales[0, :, h, w] = scales_hat[0, :, 0, 0]
        # change to channel last
        y_q = y_q.permute(0, 2, 3, 1)
        y_scales = y_scales.permute(0, 2, 3, 1)
        y_string = self.gaussian_encoder.compress(y_q, y_scales)
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_string, y_hat

    def decompress_ar(
        self,
        y_string,
        channel,
        height,
        width,
        downsample,
        kernel_size,
        context_prediction,
        params,
        entropy_parameters,
    ):
        device = next(self.parameters()).device
        padding = (kernel_size - 1) // 2

        y_size = get_downsampled_shape(height, width, downsample)
        y_height = y_size[0]
        y_width = y_size[1]

        y_hat = torch.zeros(
            (1, channel, y_height + 2 * padding, y_width + 2 * padding),
            device=params.device,
        )

        self.gaussian_encoder.set_stream(y_string)

        for h in range(y_height):
            for w in range(y_width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[0:1, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    context_prediction.weight,
                    bias=context_prediction.bias,
                )
                p = params[0:1, :, h : h + 1, w : w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)
                rv = self.gaussian_encoder.decode_stream(scales_hat)
                rv = rv.to(device)
                rv = rv + means_hat
                y_hat[
                    0, :, h + padding : h + padding + 1, w + padding : w + padding + 1
                ] = rv

        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_hat

    def compress(self, referframe, input_image):
        device = input_image.device
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)
        compressed_z_mv = torch.round(z_mv)
        mv_z_string = self.bitEstimator_z_mv.compress(compressed_z_mv)
        mv_z_size = [compressed_z_mv.size(2), compressed_z_mv.size(3)]
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)

        params_mv = self.mvpriorDecoder(mv_z_hat)
        mv_y_string, mv_y_hat = self.compress_ar(
            mvfeature, 5, self.auto_regressive_mv, params_mv, self.entropy_parameters_mv
        )

        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        temporal_prior_params = self.temporalPriorEncoder(context)
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)
        z_string = self.bitEstimator_z.compress(compressed_z)
        z_size = [compressed_z.size(2), compressed_z.size(3)]
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)

        params = self.priorDecoder(z_hat)
        y_string, y_hat = self.compress_ar(
            feature,
            5,
            self.auto_regressive,
            torch.cat((temporal_prior_params, params), dim=1),
            self.entropy_parameters,
        )

        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(
            torch.cat((recon_image_feature, context), dim=1)
        )

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = len(y_string) * 8 / pixel_num
        bpp_z = len(z_string) * 8 / pixel_num
        bpp_mv_y = len(mv_y_string) * 8 / pixel_num
        bpp_mv_z = len(mv_z_string) * 8 / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {
            "bpp_mv_y": bpp_mv_y,
            "bpp_mv_z": bpp_mv_z,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,
            "recon_image": recon_image,
            "mv_y_string": mv_y_string,
            "mv_z_string": mv_z_string,
            "y_string": y_string,
            "z_string": z_string,
        }

    def decompress(
        self, referframe, mv_y_string, mv_z_string, y_string, z_string, height, width
    ):
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        params_mv = self.mvpriorDecoder(mv_z_hat)
        mv_y_hat = self.decompress_ar(
            mv_y_string,
            self.out_channel_mv,
            height,
            width,
            16,
            5,
            self.auto_regressive_mv,
            params_mv,
            self.entropy_parameters_mv,
        )

        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        temporal_prior_params = self.temporalPriorEncoder(context)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)
        params = self.priorDecoder(z_hat)
        y_hat = self.decompress_ar(
            y_string,
            self.out_channel_M,
            height,
            width,
            16,
            5,
            self.auto_regressive,
            torch.cat((temporal_prior_params, params), dim=1),
            self.entropy_parameters,
        )
        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(
            torch.cat((recon_image_feature, context), dim=1)
        )
        recon_image = recon_image.clamp(0, 1)

        return recon_image

    def est_mv_bits(
        self, mvfeature, quant_mv, params_mv, compressed_z_mv, compress_type
    ):
        # BPP Calculations
        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)
        total_bits_mv, _ = self.feature_probs_based_sigma(
            mvfeature, means_hat_mv, scales_hat_mv, compress_type
        )
        total_bits_z_mv, _ = self.iclr18_estrate_bits_z_mv(compressed_z_mv)
        return total_bits_z_mv, total_bits_mv

    def forward(self, referframe1, referframe2, input_image, compress_type, train_type):
        assert train_type in ["memc", "memc_bpp", "recon", "full"]
        assert compress_type in ["no_compress", "train_compress", "full"]
        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]

        estmv1 = self.opticFlow(input_image, referframe1)
        estmv2 = self.opticFlow(input_image, referframe2)

        estmv = torch.cat([estmv1, estmv2], dim=1)
        # mvfeature1 = self.mvEncoder(estmv1)
        # mvfeature2 = self.mvEncoder(estmv2)
        mvfeature = self.mvEncoder(estmv)

        if compress_type == "train_compress":
            # z_mv1 = self.mvpriorEncoder(mvfeature1)
            # z_mv2 = self.mvpriorEncoder(mvfeature2)
            z_mv = self.mvpriorEncoder(mvfeature)
            # compressed_z_mv1 = self.train_quantize(z_mv1)
            # compressed_z_mv2 = self.train_quantize(z_mv2)
            compressed_z_mv = self.train_quantize(z_mv)
            # params_mv1 = self.mvpriorDecoder(compressed_z_mv1)
            # params_mv2 = self.mvpriorDecoder(compressed_z_mv2)
            params_mv = self.mvpriorDecoder(compressed_z_mv)
            # quant_mv1 = self.train_quantize(mvfeature1)
            # quant_mv2 = self.train_quantize(mvfeature2)
            quant_mv = self.train_quantize(mvfeature)
        elif compress_type == "full":
            z_mv1 = self.mvpriorEncoder(mvfeature1)
            z_mv2 = self.mvpriorEncoder(mvfeature2)
            compressed_z_mv1 = torch.round(z_mv1)
            compressed_z_mv2 = torch.round(z_mv2)
            params_mv1 = self.mvpriorDecoder(compressed_z_mv1)
            params_mv2 = self.mvpriorDecoder(compressed_z_mv2)
            quant_mv1 = torch.round(mvfeature1)
            quant_mv2 = torch.round(mvfeature2)
        else:
            raise ValueError("Unknown compress type", compress_type)

        # ctx_params_mv1 = self.auto_regressive_mv(quant_mv1)
        # gaussian_params_mv1 = self.entropy_parameters_mv(
        #     torch.cat((params_mv1, ctx_params_mv1), dim=1)
        # )
        # means_hat_mv1, scales_hat_mv1 = gaussian_params_mv1.chunk(2, 1)
        # ctx_params_mv2 = self.auto_regressive_mv(quant_mv2)
        # gaussian_params_mv2 = self.entropy_parameters_mv(
        #     torch.cat((params_mv2, ctx_params_mv2), dim=1)
        # )
        # means_hat_mv2, scales_hat_mv2 = gaussian_params_mv2.chunk(2, 1)

        # quant_mv_upsample1 = self.mvDecoder_part1(quant_mv1)
        # quant_mv_upsample2 = self.mvDecoder_part1(quant_mv2)
        quant_mv_upsample = self.mvDecoder_part1(quant_mv)
        # quant_mv_upsample_refine1 = self.mv_refine(referframe1, quant_mv_upsample1)
        # quant_mv_upsample_refine2 = self.mv_refine(referframe2, quant_mv_upsample2)
        quant_mv_upsample_refine1, quant_mv_upsample_refine2 = torch.chunk(
            quant_mv_upsample, chunks=2, dim=1
        )
        quant_mv_upsample_refine1 = self.mv_refine(
            referframe1, quant_mv_upsample_refine1
        )
        quant_mv_upsample_refine2 = self.mv_refine(
            referframe2, quant_mv_upsample_refine2
        )

        if train_type == "memc":
            pred1 = flow_warp(referframe1, quant_mv_upsample_refine1)
            pred2 = flow_warp(referframe2, quant_mv_upsample_refine2)
            return {"pred1": pred1, "pred2": pred2}
        elif train_type == "memc_bpp":
            pred1 = flow_warp(referframe1, quant_mv_upsample_refine1)
            pred2 = flow_warp(referframe2, quant_mv_upsample_refine2)
            total_bits_z_mv, total_bits_mv = self.est_mv_bits(
                mvfeature, quant_mv, params_mv, compressed_z_mv, compress_type
            )
            return {
                "pred1": pred1,
                "pred2": pred2,
                "mv_z_bpp": total_bits_z_mv / pixel_num,
                "mv_y_bpp": total_bits_mv / pixel_num,
            }

        context1 = self.motioncompensation(referframe1, quant_mv_upsample_refine1)
        context2 = self.motioncompensation(referframe2, quant_mv_upsample_refine2)

        temporal_prior_params = self.temporalPriorEncoder(
            torch.cat((context1, context2), dim=1)
        )

        feature = self.contextualEncoder(
            torch.cat((input_image, context1, context2), dim=1)
        )

        if compress_type == "train_compress":
            z = self.priorEncoder(feature)
            compressed_z = self.train_quantize(z)
            params = self.priorDecoder(compressed_z)
            feature_renorm = feature
            compressed_y_renorm = self.train_quantize(feature_renorm)
        elif compress_type == "full":
            z = self.priorEncoder(feature)
            compressed_z = torch.round(z)
            params = self.priorDecoder(compressed_z)
            feature_renorm = feature
            compressed_y_renorm = torch.round(feature_renorm)
        else:
            raise ValueError("Unknown compress type", compress_type)

        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(
            torch.cat((recon_image_feature, context1, context2), dim=1)
        )

        if train_type == "recon":
            return {"recon_image": recon_image}

        # BPP Calculations
        total_bits_z_mv, total_bits_mv = self.est_mv_bits(
            mvfeature, quant_mv, params_mv, compressed_z_mv, compress_type
        )

        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        total_bits_y, _ = self.feature_probs_based_sigma(
            feature_renorm, means_hat, scales_hat, compress_type
        )
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)

        bpp_mv_z = total_bits_z_mv / pixel_num
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {
            "bpp_mv_y": bpp_mv_y,
            "bpp_mv_z": bpp_mv_z,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,
            "recon_image": recon_image,
        }

    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict)
