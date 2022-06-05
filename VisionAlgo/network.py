import torch
import sys
from torch.utils.data import DataLoader
import numpy as np
from torch import nn, Tensor
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.roi_heads import keypointrcnn_loss as cross_entropy_kpr_loss
from torchvision.models.detection.roi_heads import keypointrcnn_inference as cross_entropy_kpr_inference
from torchvision.ops import MultiScaleRoIAlign
from torchsummary import summary
from typing import Optional, List, Dict, Tuple
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from .datasets import robotDataSets, collate_fn


def keypoints_to_heatmap(image_resolution, pointsBelief, sigma=2):
        # image size (width x height)
        # list of points to draw in a 7x2 tensor
        # the size of the point
        # returns a tensor of n_points x h x w with the belief maps
        # Input argument handling
        assert (
            len(image_resolution) == 2
        ), 'Expected "image_resolution" to have length 2, but it has length {}.'.format(
            len(image_resolution)
        )
        image_width, image_height = image_resolution
        image_transpose_resolution = (image_height, image_width)
        out = np.zeros((len(pointsBelief), image_height, image_width), dtype=np.float32)

        w = int(sigma * 2)

        for i_point, point in enumerate(pointsBelief):
            pixel_u = round(point[0])
            pixel_v = round(point[1])
            array = np.zeros(image_transpose_resolution)

            # TODO makes this dynamics so that 0,0 would generate a belief map.
            if (
                pixel_u - w >= 0
                and pixel_u + w + 1 < image_width
                and pixel_v - w >= 0
                and pixel_v + w + 1 < image_height
            ):
                for i in range(pixel_u - w, pixel_u + w + 1):
                    for j in range(pixel_v - w, pixel_v + w + 1):
                        array[j, i] = np.exp(
                            -(
                                ((i - pixel_u) ** 2 + (j - pixel_v) ** 2)
                                / (2 * (sigma ** 2))
                            )
                        )
            out[i_point] = array

        return out


def peaks_from_belief_maps(belief_map_tensor, offset_due_to_upsampling):
    # print("pfbm**************************************")

    assert (
        len(belief_map_tensor.shape) == 3
    ), "Expected belief_map_tensor to have shape [N x height x width], but it is {}.".format(
        belief_map_tensor.shape
    )

    # thresh_map_after_gaussian_filter specifies the minimum intensity in the belief map AFTER the gaussian filter.
    # with sigma = 3, a perfect heat map will have a max intensity of about 0.3. -- both in a 100x100 frame and a 400x400 frame
    thresh_map_after_gaussian_filter = 0.01
    sigma = 3

    all_peaks = []
    peak_counter = 0

    for j in range(belief_map_tensor.size()[0]):
        belief_map = belief_map_tensor[j].clone()
        map_ori = belief_map.cpu().data.numpy()

        map = gaussian_filter(map_ori, sigma=sigma)
        p = 1
        map_left = np.zeros(map.shape)
        map_left[p:, :] = map[:-p, :]
        map_right = np.zeros(map.shape)
        map_right[:-p, :] = map[p:, :]
        map_up = np.zeros(map.shape)
        map_up[:, p:] = map[:, :-p]
        map_down = np.zeros(map.shape)
        map_down[:, :-p] = map[:, p:]

        peaks_binary = np.logical_and.reduce(
            (
                map >= map_left,
                map >= map_right,
                map >= map_up,
                map >= map_down,
                map > thresh_map_after_gaussian_filter,
            )
        )
        peaks = zip(
            np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]  # x values
        )  # y values

        # Computing the weigthed average for localizing the peaks
        peaks = list(peaks)
        win = 5
        ran = win // 2
        peaks_avg = []
        for p_value in range(len(peaks)):
            p = peaks[p_value]
            weights = np.zeros((win, win))
            i_values = np.zeros((win, win))
            j_values = np.zeros((win, win))
            for i in range(-ran, ran + 1):
                for j in range(-ran, ran + 1):
                    if (
                        p[1] + i < 0
                        or p[1] + i >= map_ori.shape[0]
                        or p[0] + j < 0
                        or p[0] + j >= map_ori.shape[1]
                    ):
                        continue

                    i_values[j + ran, i + ran] = p[1] + i
                    j_values[j + ran, i + ran] = p[0] + j

                    weights[j + ran, i + ran] = map_ori[p[1] + i, p[0] + j]

            # if the weights are all zeros
            # then add the none continuous points
            try:
                peaks_avg.append(
                    (
                        np.average(j_values, weights=weights)
                        + offset_due_to_upsampling,
                        np.average(i_values, weights=weights)
                        + offset_due_to_upsampling,
                    )
                )
            except:
                peaks_avg.append(
                    (p[0] + offset_due_to_upsampling, p[1] + offset_due_to_upsampling)
                )
        # Note: Python3 doesn't support len for zip object
        peaks_len = min(
            len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0])
        )

        peaks_with_score = [
            peaks_avg[x_] + (map_ori[peaks[x_][1], peaks[x_][0]],)
            for x_ in range(len(peaks))
        ]

        id = range(peak_counter, peak_counter + peaks_len)

        peaks_with_score_and_id = [
            peaks_with_score[i] + (id[i],) for i in range(len(id))
        ]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += peaks_len

    return all_peaks


def gaussran_kpr_loss(netin_resolution, keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs):
    # keypoint_logits:Tensor  keypoint_proposals:List(Tensor)  gt_keypoints:List(Tensor)  pos_matched_idxs:List(Tensor)

    image_resolution = (keypoint_logits.shape[-2], keypoint_logits.shape[-1]) 
    
    keypoint_targets = []
    for resolution_per_image, gt_kp_in_image, midx in zip(netin_resolution, gt_keypoints, pos_matched_idxs):
        heatmaps = []
        for kp in gt_kp_in_image[midx]:
            kp[:, 0] = kp[:, 0]/resolution_per_image[0] * image_resolution[0]
            kp[:, 1] = kp[:, 1]/resolution_per_image[1] * image_resolution[1]
            heatmaps_per_image = keypoints_to_heatmap(image_resolution, kp[:, :2].cpu().numpy(), sigma=2)

            from matplotlib import pyplot as plt
            import torchvision.transforms as transforms
            temp = torch.from_numpy(heatmaps_per_image)
            plt.figure("heatmap")
            for idx in range(len(temp)):
                plt.subplot(2, 4, idx+1)
                tem_image = transforms.ToPILImage()(temp[idx])
                plt.imshow(tem_image)

            heatmaps.append(torch.from_numpy(heatmaps_per_image[None]))
        heatmaps_targets = torch.cat(heatmaps, dim=0).to(keypoint_logits.device)
        keypoint_targets.append(heatmaps_targets)
    keypoint_targets = torch.cat(keypoint_targets, dim=0).to(keypoint_logits.device)
    
    keypoint_loss = F.mse_loss(keypoint_logits, keypoint_targets)
    
    w = 10000

    return w * keypoint_loss


def gaussran_kpr_inference(keypoints_heatmaps, netin_resolution, netout_resolution, offset_due_to_upsampling=0.4395):
    keypoints_probs = []
    for ni_resolution, kps_heatmap in zip(netin_resolution, keypoints_heatmaps):
        if kps_heatmap.shape[0] == 0:
            # print("kps_heatmap shape: ",kps_heatmap.shape)
            detected_keypoint = np.array([[-999.999, -999.999, 0]])
            keypoints_per_image = np.repeat(detected_keypoint, kps_heatmap.shape[1], axis=0)[None]

        else:
            keypoints_per_image = []
            for heatmap in kps_heatmap:
                all_peaks = peaks_from_belief_maps(heatmap, offset_due_to_upsampling)
                kps = []
                for peak in all_peaks:
                    if len(peak)==1:
                        detected_keypoint = [peak[0][0], peak[0][1], 1]
                    elif len(peak)>1:
                        peak_sorted_by_score = sorted(peak, key=lambda x: x[2], reverse=True)
                        if (peak_sorted_by_score[0][2] >= 0.8):
                            detected_keypoint = [peak_sorted_by_score[0][0], peak_sorted_by_score[0][1], 1]
                        else:
                            # Can't determine -- return no detection
                            # Can't use None because we need to return it as a tensor
                            detected_keypoint = [-999.999, -999.999, 0]
                    else:
                        detected_keypoint = [-999.999, -999.999, 0]
                    # recover the keypoint
                    kp_netin = [detected_keypoint[0] / netout_resolution[0] * ni_resolution[0], detected_keypoint[1] / netout_resolution[1] * ni_resolution[1], detected_keypoint[2]]
                    kps.append(kp_netin)

                keypoints_per_image.append(kps)
        
        keypoints_per_image = np.array(keypoints_per_image, dtype=np.float32)
        keypoints_per_image = torch.from_numpy(keypoints_per_image).to(keypoints_heatmaps[0].device)

        keypoints_probs.append(keypoints_per_image)

    # return keypoints_probs, kp_scores
    return keypoints_probs


# copy from torchvision.models.detection.keypoint_rcnn.py
class KeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres1 = nn.ConvTranspose2d(
            input_features,
            256,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        # self.kps_score_lowres2 = nn.ConvTranspose2d(
        #     256,
        #     256,
        #     deconv_kernel,
        #     stride=2,
        #     padding=deconv_kernel // 2 - 1,
        # )
        self.kps_score_lowres3 = nn.ConvTranspose2d(
            256,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(self.kps_score_lowres1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres1.bias, 0)
        # nn.init.kaiming_normal_(self.kps_score_lowres2.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.kps_score_lowres2.bias, 0)
        nn.init.kaiming_normal_(self.kps_score_lowres3.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres3.bias, 0)

        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres1(x)
        # x = self.kps_score_lowres2(x)
        x = self.kps_score_lowres3(x)

        return x

        # return torch.nn.functional.interpolate(
        #     x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
        # )


class MyModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(MyModel, self).__init__()

        self.cfg = cfg
        
        if self.cfg.backbone_type == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=self.cfg.backbone_pretrained)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool

            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

            # upconvolution and final layer
            BN_MOMENTUM = 0.1
            self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=4,
            #     stride=2,
            #     padding=1,
            #     output_padding=0,
            # ),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=4,
            #     stride=2,
            #     padding=1,
            #     output_padding=0,
            # ),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, self.cfg.backbone_n_keypoints, kernel_size=1, stride=1),
        )
        

        elif self.cfg.backbone_type == "keypointrcnn_resnet50_fpn":
            anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
            self.keypoint_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                            pretrained_backbone=self.cfg.backbone_pretrained,
                                                                            num_keypoints=self.cfg.backbone_n_keypoints,
                                                                            num_classes=2, # background and robot
                                                                            rpn_anchor_generator=anchor_generator)
            

        elif self.cfg.backbone_type == "keypoint_rcnn_finetune":
            # box
            representation_size = 1024
            box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                representation_size,
                num_classes=2)
            
            # keypoint
            keypoint_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=56,
                sampling_ratio=2)

            backbone_out_channels = 256
            keypoint_layers = tuple(256 for _ in range(4))
            keypoint_head = torchvision.models.detection.keypoint_rcnn.KeypointRCNNHeads(backbone_out_channels, keypoint_layers)

            keypoint_dim_reduced = 256  # == keypoint_layers[-1]
            keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, self.cfg.backbone_n_keypoints)

            anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
            keypoint_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                            pretrained_backbone=self.cfg.backbone_pretrained,
                                                                            num_keypoints=None,
                                                                            num_classes=None, # background and robot
                                                                            # min_size=(240, 480, 640, 760, 800),
                                                                            # max_size=(1),
                                                                            # rpn_pre_nms_top_n_train=1000, rpn_pre_nms_top_n_test=500,
                                                                            # rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=500,

                                                                            # rpn_pre_nms_top_n_train=1000, rpn_pre_nms_top_n_test=500,
                                                                            # rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=500,   # 05_02_17

                                                                            rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                                                                            rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,

                                                                            rpn_anchor_generator=anchor_generator,
                                                                            box_predictor=box_predictor,
                                                                            keypoint_roi_pool=keypoint_roi_pool,
                                                                            keypoint_head=keypoint_head,
                                                                            keypoint_predictor=keypoint_predictor,
                                                                            )
            self.transform = keypoint_rcnn.transform
            self.backbone = keypoint_rcnn.backbone
            self.rpn = keypoint_rcnn.rpn
            self.roi_heads = keypoint_rcnn.roi_heads

            self.box_roi_pool = self.roi_heads.box_roi_pool
            self.box_head = self.roi_heads.box_head
            self.box_predictor = self.roi_heads.box_predictor

            self.keypoint_roi_pool = self.roi_heads.keypoint_roi_pool
            self.keypoint_head = self.roi_heads.keypoint_head
            self.keypoint_predictor = self.roi_heads.keypoint_predictor

        else:
            print("the MyModel backbone_type can't accepted")
            sys.exit(1)


    def forward(self, images, targets=None):

        if self.cfg.backbone_type == "resnet50":
            x = self.conv1(images)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # x = self.layer4(x)

            x = self.upsample(x)
            return x

        elif self.cfg.backbone_type == "keypointrcnn_resnet50_fpn":
            
            return self.keypoint_rcnn(images, targets)
        
        elif self.cfg.backbone_type == "keypoint_rcnn_finetune":

            original_image_sizes = list((image.shape[1], image.shape[2]) for image in images)
            images, targets = self.transform(images, targets)
            features = self.backbone(images.tensors)
            proposals, proposal_losses = self.rpn(images, features, targets)

            if self.training:
                proposals, matched_idxs, labels, regression_targets = self.roi_heads.select_training_samples(proposals, targets)
            else:
                labels = None
                regression_targets = None
                matched_idxs = None

            # boxes
            box_features = self.box_roi_pool(features, proposals, images.image_sizes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)

            result: List[Dict[str, torch.Tensor]] = []
            losses = {}
            if self.training:
                assert labels is not None and regression_targets is not None
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets)
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg
                }
            else:
                boxes, scores, labels = self.roi_heads.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                        }
                    )


            # keypoints
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                # print('num_images:  ', num_images)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, images.image_sizes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                
                gt_keypoints = [t["keypoints"] for t in targets]
                
                if self.cfg.output_type == 'gaussian heatmap':
                    rcnn_loss_keypoint = gaussran_kpr_loss(
                        images.image_sizes,
                        keypoint_logits, keypoint_proposals,
                        gt_keypoints, pos_matched_idxs)
                elif self.cfg.output_type == 'cross_entropy heatmap':
                    rcnn_loss_keypoint = cross_entropy_kpr_loss(
                        keypoint_logits, keypoint_proposals,
                        gt_keypoints, pos_matched_idxs)
                else:
                    sys.exit(1)

                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                if self.cfg.output_type == 'gaussian heatmap':
                    boxes_per_image = [box.size(0) for box in keypoint_proposals]
                    keypoints_heatmaps = keypoint_logits.split(boxes_per_image, dim=0)

                    keypoints_probs = gaussran_kpr_inference(keypoints_heatmaps, images.image_sizes, keypoint_logits.shape[-2:])
                    for kps_heatmap, keypoint_prob, r in zip(keypoints_heatmaps, keypoints_probs, result):
                        r["keypoints"] = keypoint_prob
                        r["heatmaps"] = kps_heatmap

                elif self.cfg.output_type == 'cross_entropy heatmap':
                    keypoints_probs, kp_scores = cross_entropy_kpr_inference(keypoint_logits, keypoint_proposals)
                    for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                        r["keypoints"] = keypoint_prob
                        r["keypoints_scores"] = kps
                else:
                    sys.exit(1)

            losses.update(loss_keypoint)

            total_losses = {}
            total_losses.update(losses)
            total_losses.update(proposal_losses)
            
            detections = self.transform.postprocess(result, images.image_sizes, original_image_sizes)
            
            return total_losses, detections

        else:
            print("the MyModel backbone_type can't accepted")
            sys.exit(1)


if __name__ == "__main__":
    from config import set_config
    dataPath = "/home/lihua/Desktop/Datasets/DREAM/synthetic/panda_synth_train_dr"

    # ################"keypointrcnn_resnet50_fpn"####################
    # keypointrcnn = MyModel("keypointrcnn_resnet50_fpn", 7, pretrained=True).to(device)
    # keypointrcnn.eval()
    # image = torch.rand(3, 480, 640).to(device)
    # target = {}
    # target['boxes'] = torch.tensor([[1, 2, 10, 20]]).to(device)
    # target['labels'] = torch.tensor([0]).to(device)
    # target['keypoints'] = torch.rand([1, 7, 3]).to(device)

    # result2 = keypointrcnn([image, image], [target, target])

    # ##########################"resnet"############################
    # resnet = MyModel("resnet50", 7, pretrained=True).to(device)
    # resnet.eval()
    # image = torch.rand(1, 3, 400, 400).to(device)
    # result = resnet(image)

    # summary(resnet, [(3, 400, 400)])

    ###################"keypointrcnn_finetune"#####################
    rds_rcnn_finetune = robotDataSets(dataPath, "panda", "keypoint_rcnn_finetune")
    train_loader = DataLoader(dataset=rds_rcnn_finetune, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn)
    rds_num_data = len(rds_rcnn_finetune)
    # print("the num of image:", rds_num_data)
    n_train_data = int(round(rds_num_data * 0.8))
    n_valid_data = rds_num_data - n_train_data
    rds_train, rds_test = torch.utils.data.random_split(rds_rcnn_finetune, [n_train_data, n_valid_data])

    train_loader = DataLoader(dataset=rds_train, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=rds_test, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)

    iterator = iter(test_loader)
    batch = next(iterator)

    cfg = set_config(True)
    torch.cuda.set_device(cfg.gpu_id[0])
    
    keypointrcnn_finetune = MyModel(cfg).to(cfg.device)
    # keypointrcnn_finetune.eval()

    images = list(image.to(cfg.device) for image in batch[0])
    labels = [{k: v.to(cfg.device) for k, v in t.items()} for t in batch[1]]

    losses, result1 = keypointrcnn_finetune(images, labels)

    # image = torch.rand(3, 480, 640).to(device)
    # target = {}
    # target['boxes'] = torch.tensor([[1, 2, 10, 20]]).to(device)
    # target['labels'] = torch.tensor([1]).to(device)
    # target['keypoints'] = torch.rand([1, 7, 3]).to(device)

    # result2 = keypointrcnn_finetune([image, image], [target, target])

    

    print("debug line")


