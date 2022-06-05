from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import numpy as np
from VisionAlgo.datasets import robotDataSets
from VisionAlgo.network import MyModel
import torchvision.transforms as transforms
from VisionAlgo.config import set_config
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import DataLoader

# Code adapted from code originally written by Jon Tremblay
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

def keypoint_rcnn_inference(pretrained_model_path, network_type, test_loader):
    cfg = set_config(True, network_type)

    # rds = robotDataSets(cfg.data_path, cfg.robot_type, cfg.network_type)

    model = MyModel(cfg)
    
    if cfg.device.type == 'cuda':
        torch.cuda.set_device(cfg.gpu_id[0])
        # model = nn.DataParallel(model, device_ids=[gpu for gpu in cfg.gpu_id])
        # model.to(f'cuda:{model.device_ids[0]}')
        model.cuda()
        model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:'+str(cfg.gpu_id[0])))
    else:
        model.load_state_dict(torch.load(pretrained_model_path))

    model.eval()
    with torch.no_grad():
        total_oks = []
        for i, (images, labels) in enumerate(test_loader):
            images = list(image.to(f'cuda:{cfg.gpu_id[0]}') for image in images)
            labels = [{k: v.to(f'cuda:{cfg.gpu_id[0]}') for k, v in t.items()} for t in labels]

            # forward
            detections = model(images, labels)

            metrics = calculate_metrics(detections, labels)
            total_oks.append(metrics['oks'])

            if i % 100 == 0:
                    print("Total Validing OKS: [{}]".format(sum(total_oks)/len(total_oks)))

        return sum(total_oks)/len(total_oks)


def keypoint_rcnn_finetune_inference(pretrained_model_path, network_type, raw_image):
    cfg = set_config(True, network_type)

    model = MyModel(cfg)
    model.eval()
    
    if cfg.device.type == 'cuda':
        # if len(cfg.gpu_id)==1:
        #     torch.cuda.set_device(cfg.gpu_id[0])
        #     model.cuda()
        #     model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:'+str(cfg.gpu_id[0])))
        # else:
            model = torch.nn.DataParallel(model, device_ids=[cfg.gpu_id[0]])
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:'+str(cfg.gpu_id[0])))
            model.to(f'cuda:{model.device_ids[0]}')
    else:
        model.load_state_dict(torch.load(pretrained_model_path))

    images = transforms.ToTensor()(raw_image)
    images = torch.unsqueeze(images, 0)
    images = list(image.to(cfg.device) for image in images)
    # labels = [{k: v.to(cfg.device) for k, v in t.items()} for t in labels]

    losses, result = model(images)

    keypointsNameList = ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link7', 'panda_hand', 'panda_link0', 'panda_link6']

    idx = torch.argmax(result[0]['scores'])
    if result[0]['labels'][idx] == 0:
        return {'boxes': None, 'keypoints': None, 'heatmaps': None}, keypointsNameList
    else:
        bbox = torch.round(torch.squeeze(result[0]['boxes'][idx], 0)).cpu().detach().numpy()
        keypoints = result[0]['keypoints'][idx].cpu().detach().numpy()
        # heatmaps = result[0]['heatmaps'][idx].cpu().detach().numpy()
        heatmaps = None
        return {'boxes': bbox, 'keypoints':keypoints, 'heatmaps': heatmaps}, keypointsNameList

def calculate_metrics(detections, targets):
    '''
    dst_keypoint: List(Tensor(N, 7, 3))
    gt_keypoints: List(Tensor(1, 7, 3))
    gt_roi: List(Tensor(1, 4))
    '''
    result = {}
    
    # object keypoint similarity
    similarity = []
    for detection, target in zip(detections, targets):

        dst_keypoints, dst_scores, gt_keypoints, gt_roi = detection['keypoints'], detection['scores'], target['keypoints'],  target['boxes']

        if dst_keypoints.shape[0] == 0:
            similarity.append(0)
        else:    
            score_index = torch.argmax(dst_scores)
            _, vis_index = torch.where(gt_keypoints[:, :, -1] == 1)

            # area
            src_area = (gt_roi[:, 2] - gt_roi[:, 0] + 1) * (gt_roi[:, 3] - gt_roi[:, 1] + 1)

            # measure the per-keypoint distance if keypoints visible
            dx = dst_keypoints[score_index, vis_index, 0] - gt_keypoints[0, vis_index, 0]
            dy = dst_keypoints[score_index, vis_index, 1] - gt_keypoints[0, vis_index, 1]

            e = (dx**2 + dy**2) / (src_area + np.spacing(1))
            e = torch.sum(torch.exp(-e), dim=0) / e.shape[0]
            similarity.append(e.item())
    
    result['oks'] = sum(similarity)/len(similarity)

    return result


def collate_fn(batch):
        return tuple(zip(*batch))

        
if __name__ == "__main__":
    # ##############################################"resnet"##############################################
    # pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/resnet50220407_210209.pkl"
    # # raw_image_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/img/PandaRobotPose.jpg"
    # # raw_image = Image.open(raw_image_path).convert('RGB')
    # keypoins_heatmap_batch, recovered_keypoints_batch, keypointsNameList = resnet_inference(pretrained_model_path)
    

    ##############################################"keypoint_rcnn"##############################################
    # data_path = "/home/lihua/Desktop/Datasets/DREAM/real/panda-3cam_kinect360"
    data_path = "/home/lihua/Desktop/Datasets/DREAM/synthetic/panda_synth_test_dr"
    pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/keypoint_rcnn_04_21_15_49.pkl"
    
    rds = robotDataSets(data_path, "panda", "keypoint_rcnn")
    test_loader = DataLoader(dataset=rds, batch_size=5, shuffle=False, num_workers=8, collate_fn=collate_fn)
    # raw_image_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/img/PandaRobotPose.jpg"
    
    result = keypoint_rcnn_inference(pretrained_model_path,"keypoint_rcnn", test_loader)
    
    print('the average oks is:', result)
    