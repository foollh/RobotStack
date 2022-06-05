from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from VisionAlgo.datasets import robotDataSets
from VisionAlgo.network import MyModel
import torchvision.transforms as transforms
from VisionAlgo.config import set_config
from scipy.ndimage.filters import gaussian_filter

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

def resnet_inference(pretrained_model_path, raw_image):
    cfg = set_config(True, network_type='resnet')

    rds = robotDataSets(cfg.data_path, cfg.robot_type, cfg.network_type)

    with torch.no_grad():
        model = MyModel(cfg)
        # if cfg.device.type == 'cuda':
        #     model.cuda()
        model.load_state_dict(torch.load(pretrained_model_path))

        preprocessed_image = rds.pre_inference(raw_image)
        
        # if cfg.device.type == 'cuda':
        #     preprocessed_image.to(cfg.device)
        #     model.cuda()
    
        keypoins_heatmap_batch = model(preprocessed_image)
        
        recovered_keypoints_batch = []
        for keypoints_heatmap in keypoins_heatmap_batch:
            all_peaks = peaks_from_belief_maps(keypoints_heatmap, 0.4395)
            resize_keypoints = []
            for peak in all_peaks:
                if len(peak)==1:
                    detected_keypoints = [peak[0][0], peak[0][1]]
                elif len(peak)>1:
                    peak_sorted_by_score = sorted(peak, key=lambda x: x[2], reverse=True)
                    if (peak_sorted_by_score[0][2]-peak_sorted_by_score[1][2] >= 0.25):
                        detected_keypoints = [peak_sorted_by_score[0][0], peak_sorted_by_score[0][1],]
                    else:
                        # Can't determine -- return no detection
                        # Can't use None because we need to return it as a tensor
                        detected_keypoints = [-999.999, -999.999]
                else:
                    detected_keypoints = [-999.999, -999.999]
                # recover the keypoints
                kp_netin = [detected_keypoints[0] / 200 * 640, detected_keypoints[1] / 200 * 480, ]
                resize_keypoints.append(kp_netin)

            recovered_keypoints_batch.append(resize_keypoints)

        recovered_keypoints_batch = torch.tensor(recovered_keypoints_batch).float()

    keypointsNameList = ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link7', 'panda_hand', 'panda_link0', 'panda_link6']
    
    return [keypoins_heatmap_batch, recovered_keypoints_batch, keypointsNameList]


def keypoint_rcnn_inference(pretrained_model_path, network_type, raw_image):
    cfg = set_config(True, network_type)

    # rds = robotDataSets(cfg.data_path, cfg.robot_type, cfg.network_type)

    model = MyModel(cfg)
    model.eval()
    
    if cfg.device.type == 'cuda':
        torch.cuda.set_device(cfg.gpu_id[0])
        model.cuda()
        model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:'+str(cfg.gpu_id[0])))
    else:
        model.load_state_dict(torch.load(pretrained_model_path))

    images = transforms.ToTensor()(raw_image)
    images = torch.unsqueeze(images, 0)
    images = list(image.to(cfg.device) for image in images)
    # labels = [{k: v.to(cfg.device) for k, v in t.items()} for t in labels]

    result = model(images)

    keypointsNameList = ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link7', 'panda_hand', 'panda_link0', 'panda_link6']

    idx = torch.argmax(result[0]['scores'])
    if result[0]['labels'][idx] == 0:
        return {'boxes': None, 'keypoints': None}, keypointsNameList
    else:
        bbox = torch.round(torch.squeeze(result[0]['boxes'][idx], 0)).cpu().detach().numpy()
        keypoints = result[0]['keypoints'][idx].cpu().detach().numpy()
        return {'boxes': bbox, 'keypoints':keypoints}, keypointsNameList

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


if __name__ == "__main__":
    # ##############################################"resnet"##############################################
    # pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/resnet50220407_210209.pkl"
    # raw_image_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/img/PandaRobotPose.jpg"
    # raw_image = Image.open(raw_image_path).convert('RGB')
    # keypoins_heatmap_batch, recovered_keypoints_batch, keypointsNameList = resnet_inference(pretrained_model_path, raw_image)
    # print("##### Detected keypoints #####")
    # print("{}".format(recovered_keypoints_batch))

    # plt.figure("image")
    # draw = ImageDraw.Draw(raw_image)
    # ttf = ImageFont.load_default()
    # radius = 3
    # for keypoints in recovered_keypoints_batch:
    #     for idx, kp in enumerate(keypoints.numpy()):
    #         kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
    #         kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
    #         # print("the drawn keypoints: \n {} {}".format(kp_left_up, kp_right_down))
    #         draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(255,0,0))
    #         draw.ellipse((kp_left_up, kp_right_down), fill=(0, 255, 0))
    # plt.imshow(raw_image)

    # plt.figure("heatmap")
    # for idx in range(len(keypoins_heatmap_batch[0])):
    #     plt.subplot(2, 4, idx+1)
    #     plt.imshow(transforms.ToPILImage()(keypoins_heatmap_batch[0][idx]))
    # plt.show()

    # ##############################################"keypoint_rcnn"##############################################
    # # pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/keypoint_rcnn_04_19_17_54.pkl"
    # pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/keypoint_rcnn_04_21_15_49.pkl"
    
    # raw_image_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/img/PandaRobotPose.jpg"
    
    # raw_image = Image.open(raw_image_path).convert('RGB')
    # result, keypointsNameList = keypoint_rcnn_inference(pretrained_model_path,"keypoint_rcnn", raw_image)
    # # print("##### Detected keypoints #####")
    # # print("{}".format(recovered_keypoints_batch))

    # plt.figure("image")
    # draw = ImageDraw.Draw(raw_image)
    # ttf = ImageFont.load_default()
    # radius = 3

    # # boxes
    # bbox = result['boxes']
    # draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
    # # keypoints
    # keypoints = result['keypoints']
    # for idx, kp in enumerate(keypoints):
    #     kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
    #     kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
    #     # print("the drawn keypoints: \n {} {}".format(kp_left_up, kp_right_down))
    #     # draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(255,0,0))
    #     draw.ellipse((kp_left_up, kp_right_down), fill=(255, 0, 0))
    
    # # gt
    # gt_image_keypoints = np.array([[160.00001669, 399.99998649],
    #                         [160.00001669, 266.79999801],
    #                         [218.97280094, 155.00034643],
    #                         [248.1605716,  170.39626814],
    #                         [405.21453259, 166.28291996],
    #                         [434.26901511, 146.40847851],
    #                         [458.43489798, 181.73532361]])
    # for idx, kp in enumerate(gt_image_keypoints):
    #     kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
    #     kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
    #     draw.ellipse((kp_left_up, kp_right_down), fill=(0, 255, 0))

    # plt.imshow(raw_image)

    # # plt.figure("heatmap")
    # # for idx in range(len(result[0])):
    # #     plt.subplot(2, 4, idx+1)
    # #     plt.imshow(transforms.ToPILImage()(result[0][idx]))
    
    # plt.show()
    
    ##############################################" keypoint_rcnn_finetune "##############################################
    pretrained_model_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/VisionAlgo/trained_models/keypoint_rcnn_finetune_05_05_14_42.pkl"
    # raw_image_path = "/home/lihua/Desktop/Repositories/Project/RobotStack/img/PandaRobotPose.jpg"
    raw_image_path = "/home/lihua/Desktop/Datasets/DREAM/real/panda-3cam_kinect360/000358.rgb.jpg"
    # raw_image_path = "/home/lihua/Desktop/Datasets/DREAM/synthetic/panda_synth_test_photo/000248.rgb.jpg"

    
    raw_image = Image.open(raw_image_path).convert('RGB')
    result, keypointsNameList = keypoint_rcnn_finetune_inference(pretrained_model_path,"keypoint_rcnn_finetune", raw_image)
    print("##### Detected keypoints #####")
    print("{}".format(result['keypoints']))

    # # heatmaps
    # plt.figure("heatmap")
    # for idx in range(len(result['heatmaps'])):
    #     plt.subplot(2, 4, idx+1)
    #     plt.imshow(transforms.ToPILImage()(result['heatmaps'][idx]))

    plt.figure("image")
    draw = ImageDraw.Draw(raw_image)
    ttf = ImageFont.load_default()
    radius = 3

    # boxes
    bbox = result['boxes']
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)

    # keypoints
    keypoints = result['keypoints']
    for idx, kp in enumerate(keypoints):
        kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
        kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
        # print("the drawn keypoints: \n {} {}".format(kp_left_up, kp_right_down))
        # draw.text(kp_right_down, keypointsNameList[idx], font=ttf, fill=(255,0,0))
        draw.ellipse((kp_left_up, kp_right_down), fill=(255, 0, 0))
    
    raw_image.save("./img/temp2.jpg")

    # # gt
    # gt_image_keypoints = np.array([[120.00002384, 439.99998013],
    #                                 [120.00002384, 306.79999165],
    #                                 [178.97284485, 195.00030331],
    #                                 [208.16037461, 210.39646593],
    #                                 [365.21335929, 206.28409405],
    #                                 [394.26767901, 186.4098154 ],
    #                                 [418.43334365, 221.73687874]])

    # for idx, kp in enumerate(gt_image_keypoints):
    #     kp_left_up = (round(kp[0]-radius), round(kp[1]-radius))
    #     kp_right_down = (round(kp[0]+radius), round(kp[1]+radius))
    #     draw.ellipse((kp_left_up, kp_right_down), fill=(0, 255, 0))

    plt.imshow(raw_image)

    
    plt.show()