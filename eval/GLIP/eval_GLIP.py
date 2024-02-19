import matplotlib.pyplot as plt
from typing import List, Tuple
# import matplotlib.pylab as pylab
# pylab.rcParams['figure.figsize'] = 20, 12
import argparse
# from google.colab.patches import cv2_imshow
import cv2
import json
import requests
import os
import torch
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from tqdm import tqdm
import torchvision.ops as ops


def bbox_iou(box1: float, box2: float, return_intersection_area: bool = False,
             return_intersection_area_as_bbox: bool = False):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection_area = max(x_max - x_min, 0) * max(y_max - y_min, 0)
    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area

    return_value = iou
    if return_intersection_area:
        if not isinstance(return_value, list):
            return_value = [return_value]
        return_value.append(intersection_area)
    if return_intersection_area_as_bbox:
        if not isinstance(return_value, list):
            return_value = [return_value]
        return_value.append([x_min, y_min, x_max, y_max])

    return return_value


def find_best_matching_tensor(input_tensor: torch.Tensor, tensors: List[torch.Tensor],
                              min_iou_threshold: float = 0.5) -> torch.Tensor:
    best_matching_tensor = None
    best_idx = None
    best_iou = -1.0

    for idx, tensor in enumerate(tensors):
        iou = bbox_iou(input_tensor, tensor)
        if iou > best_iou and iou >= min_iou_threshold:
            best_iou = iou
            best_matching_tensor = tensor
            best_idx = idx
        else:
            best_iou = iou
    if best_idx is None:
        print(f"iou: {best_iou} while min iou threshold is {min_iou_threshold}")

    return best_matching_tensor, best_idx


def filter_similar_boxes(bboxes, labels, scores, iou_threshold=0.8):
    # Initialize lists to store the filtered bounding boxes and their corresponding labels
    filtered_bboxes = []
    filtered_labels = []
    filtered_scores = []
    seen_boxes = set()

    # Iterate through the bounding boxes and keep only those with the highest score for each similar box group
    for i in range(len(bboxes)):
        box_i = bboxes[i]
        label_i = labels[i]
        score_i = scores[i]

        skip_box = False
        for seen_box, seen_label in seen_boxes:
            iou = bbox_iou(box_i, seen_box)
            if iou > iou_threshold:
                skip_box = True
                if score_i > scores[seen_label]:
                    # Update the seen box if the current box has a higher score
                    seen_boxes.remove((seen_box, seen_label))
                    seen_boxes.add((box_i, i))
                break

        if not skip_box:
            seen_boxes.add((box_i, i))

    # Extract the filtered boxes and labels from the seen_boxes set
    for box, label_idx in seen_boxes:
        filtered_bboxes.append(bboxes[label_idx])
        filtered_labels.append(labels[label_idx])
        filtered_scores.append(scores[label_idx])

    # Convert the lists to torch tensors
    filtered_bboxes = torch.stack(filtered_bboxes)
    filtered_labels = torch.tensor(filtered_labels)
    filtered_scores = torch.stack(filtered_scores)

    return filtered_bboxes, filtered_labels, filtered_scores


def position_subtitle(location_active_speaker, position_height_percent=80, height=1080, width=1920):
    x1_start, y1_start, x1_end, y1_end = location_active_speaker

    # Calculate the middle points of the bounding boxes
    x1_middle = (x1_start + x1_end) / 2
    # y1_middle = (y1_start + y1_end) / 2
    if isinstance(x1_middle, torch.Tensor):
        # Convert the tensor to a Python list
        x1_middle = x1_middle.tolist()

    x = x1_middle
    y = (position_height_percent * height) / 100

    return [x - 15, y - 15, x + 15, y + 15]


def evaluate_subtitles_with_json_target(path_json: str, path_images: str, glip_own, caption: str = "spokesman",
                                        print_single_result: bool = True, threshold_predictions: float = 0.3,
                                        threshold_overlap: float = 0.5, transform_annotation_bboxes_format: bool = True,
                                        normalize: bool = True, position_height: float = 80):
    f = open(path_json)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    # Iterating through the json
    # list

    categories = {}
    for pdict in data["categories"]:
        if pdict["name"].lower() in caption.lower() or pdict["name"].lower() == "subtitle-position":
            categories[pdict["id"]] = pdict["name"].lower()
    print(f"categories: {categories}")

    final_dict = {}
    for pdict in data["annotations"]:
        if pdict["category_id"] in categories.keys():
            file_name = next((adict["file_name"] for adict in data["images"] if adict['id'] == pdict["image_id"]), -1)
            if str(file_name) == str(-1):
                file_name = next((adict["file_name"] for adict in data["images"] if
                                  int(adict['file_name'].replace(".jpg", "").split("-")[-1]) == int(pdict["image_id"])),
                                 -1)
                image_width = next((adict["width"] for adict in data["images"] if
                                    int(adict['file_name'].replace(".jpg", "").split("-")[-1]) == int(
                                        pdict["image_id"])), -1)
                image_height = next((adict["height"] for adict in data["images"] if
                                     int(adict['file_name'].replace(".jpg", "").split("-")[-1]) == int(
                                         pdict["image_id"])), -1)
            else:
                image_width = next((adict["width"] for adict in data["images"] if adict['id'] == pdict["image_id"]), -1)
                image_height = next((adict["height"] for adict in data["images"] if adict['id'] == pdict["image_id"]),
                                    -1)
            if file_name in final_dict:
                final_dict[file_name]["labels"].append(pdict["category_id"])
                final_dict[file_name]["bboxes"].extend(
                    [[i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                     [pdict["bbox"]]])
                # final_dict[file_name]["width"].append(image_width)
                # final_dict[file_name]["height"].append(image_height)
                # final_dict [final_dict[file_name], pdict["category_id"], pdict["bbox"]]
            else:
                final_dict[file_name] = {"labels": [pdict["category_id"]], "bboxes": [
                    [i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                    [pdict["bbox"]]], "height": image_height, "width": image_width}

    print(f"final_dict: {final_dict}")
    files = {}
    for pdict in data["images"]:
        files[pdict["file_name"].replace(":", "_")] = pdict["id"]

    print(f"files: {files}")
    distances = []
    for filename in tqdm(os.listdir(path_images)):
        print(f"filename: {filename}")
        if filename.endswith("jpg"):
            # if int(filename.split(".")[0].split("-")[-1]) in final_dict:
            if filename.replace(":", "_").split("-")[-1] in files or filename in files or filename.split("-")[
                -1] in files:
                print("execute logic")
                image = Image.open(path_images + r"/" + filename).convert("RGB")
                image = np.array(image)  # [:, :, [2, 1, 0]]
                # caption = 'spokesman'
                result, top_predictions = glip_own.run_on_web_image(image, caption, threshold_predictions)
                scores = top_predictions.get_field("scores")
                bboxes = top_predictions.bbox
                labels = top_predictions.get_field("labels")
                # entities = glip_own.entities

                if len(caption.replace(" ", "").split(".")) == 1:
                    bboxes, labels, scores = remove_indices_below_max(bboxes, scores, labels)
                else:
                    bboxes, labels, scores = filter_similar_boxes(bboxes, labels, scores)
                # print(bboxes)
                # print(final_dict[int(filename.split(".")[0])])
                # print(f"filename {filename} in final_dict: {filename in final_dict}")
                if filename.split("-")[-1] in final_dict:
                    filename = filename.split("-")[-1]
                if filename in final_dict:
                    if len(bboxes) == 1:
                        # temp = torch.tensor(final_dict[filename]["bboxes"])
                        temp_bbox = final_dict[filename]["bboxes"][0]
                        # print(f"subtitle_position_target: {temp_bbox}")
                        subtitle_position = position_subtitle(bboxes[0], position_height_percent=position_height)
                        # print(f"subtitle_position_predicted: {subtitle_position}")
                        print(final_dict[filename]["bboxes"][0])
                        print(f"new subtitle_position: {torch.tensor(subtitle_position)}")
                        distance_temp = calculate_distance_between_bounding_boxes(final_dict[filename]["bboxes"][0],
                                                                                  torch.tensor(subtitle_position),
                                                                                  picture_width=torch.tensor(final_dict
                                                                                                             [filename]
                                                                                                             [
                                                                                                                 "width"]),
                                                                                  picture_height=torch.tensor(
                                                                                      final_dict
                                                                                      [filename]
                                                                                      ["height"]),
                                                                                  normalize=normalize)
                        distances.append(distance_temp)
                        # print(distances)
                        # image_pil = Image.open(path_images + r"/" + filename).convert("RGB")
                        # image_array = np.array(image_pil)
                        # Extract the bounding box coordinates
                        # x_start, y_start, x_end, y_end = final_dict[filename]["bboxes"][0]
                        # x_start, y_start, x_end, y_end = int(x_start), int(y_start), int(x_end), int(y_end)
                        # Create a mask for the bounding box
                        # mask = np.zeros_like(image_array)
                        # image_array = cv2.rectangle(image_array, (x_start, y_start), (x_end, y_end), (255, 255, 255))
                        # Inpaint the region inside the bounding box
                        # inpainted_image = cv2.inpaint(image_array, mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_TELEA)

                        # x_start, y_start, x_end, y_end = subtitle_position
                        # x_start, y_start, x_end, y_end = int(x_start), int(y_start), int(x_end), int(y_end)
                        # x_start = x_start - 3
                        # y_start = y_start - 3
                        # x_length = 6
                        # y_length = 6

                        # mask = np.zeros_like(image_array)
                        # image_array = cv2.rectangle(image_array, (x_start, y_start), (x_end, y_end), (255, 255, 255))
                        # Inpaint the region inside the bounding box
                        # inpainted_image = cv2.inpaint(image_array, mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                        # cv2.imwrite('inpainted_image_with_bbox.jpg', image_array)
                        # print(os.path.abspath('inpainted_image_with_bbox.jpg'))
                        # break
                    else:
                        results = []
                        for bbox in bboxes:
                            subtitle_position = position_subtitle(bbox)
                            results.append(
                                calculate_distance_between_bounding_boxes(bbox, torch.tensor(subtitle_position),
                                                                          picture_width=torch.tensor(final_dict
                                                                                                     [filename]
                                                                                                     ["width"]),
                                                                          picture_height=torch.tensor(final_dict
                                                                                                      [filename]
                                                                                                      ["height"]),
                                                                          normalize=normalize))
                        distances.append(mean(results))
                else:
                    if normalize:
                        distances.append(np.nan)
                    else:
                        distances.append(np.nan)

    # print("precision: " + str(true_positive / (true_positive + false_positive)))
    # print("recall: " + str(true_positive / (true_positive + false_negative)))
    return {"distances": sum(distances) / len(distances)}


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.show()


def imsave(img, path):
    import matplotlib.image as mpimg
    print(path)
    mpimg.imsave(path, img[:, :, [2, 1, 0]])


def initialize(config_file: str, weight_file: str):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    return GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.5,
        show_mask_heatmaps=False
    )


def evaluate_json(path_json: str, path_images: str, glip_own, caption: str = "spokesman. listener",
                  print_single_result: bool = True, threshold_predictions: float = 0.3, threshold_overlap: float = 0.5,
                  transform_annotation_bboxes_format: bool = True):
    # assert "." not in caption, "Yet only active_speaker-single_and_voting captions are supported"
    # Opening JSON file
    f = open(path_json)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # transform_bboxes_length_to_point(data)
    # Closing file
    f.close()
    # Iterating through the json
    # list

    categories = {}
    for pdict in data["categories"]:
        if pdict["name"].lower() in caption.lower():
            categories[pdict["id"]] = pdict["name"].lower()

    final_dict = {}
    for pdict in data["annotations"]:
        if pdict["category_id"] in categories.keys():
            file_name = next((adict["file_name"] for adict in data["images"] if adict['id'] == pdict["image_id"]), -1)
            if file_name in final_dict:
                final_dict[file_name]["labels"].append(pdict["category_id"])
                final_dict[file_name]["bboxes"].extend(
                    [[i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                     [pdict["bbox"]]])
                # final_dict [final_dict[file_name], pdict["category_id"], pdict["bbox"]]
            else:
                final_dict[file_name] = {"labels": [pdict["category_id"]], "bboxes": [
                    [i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                    [pdict["bbox"]]]}

    files = {}
    for pdict in data["images"]:
        files[pdict["file_name"].replace(":", "_")] = pdict["id"]

    print(f"final_dict: {final_dict}")
    print(f"files: {files}")
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    counter_all = 0

    for filename in tqdm(os.listdir(path_images)):
        if filename.endswith("jpg"):
            print(filename)
            # if int(filename.split(".")[0].split("-")[-1]) in final_dict:
            if filename.replace(":", "_") in files or filename in files:
                image = Image.open(path_images + r"/" + filename).convert("RGB")

                # image_draw = Image.open(path_images + r"/" + filename).convert("RGB")
                # draw = ImageDraw.Draw(image_draw)

                image = np.array(image)  # [:, :, [2, 1, 0]]
                # caption = 'spokesman'
                result, top_predictions = glip_own.run_on_web_image(image, caption, threshold_predictions)

                scores = top_predictions.get_field("scores")
                bboxes = top_predictions.bbox
                labels = top_predictions.get_field("labels")
                entities = glip_own.entities

                if print_single_result:
                    print(f"All boxes: {bboxes}")
                    print(f"All labels: {labels}")
                    print(f"All scores: {scores}")

                if bboxes.numel() != 0:
                    if caption == "spokesman":
                        try:
                            bboxes, labels, scores = remove_indices_below_max(bboxes, scores, labels)
                            # Define the bounding box coordinates (replace these with your actual coordinates)
                            box_coordinates = bboxes[0].tolist()  # Format: [(x1, y1), (x2, y2)]
                            # print(f"box_coordinates {box_coordinates}")
                            # Draw the bounding box on the image
                            # draw.rectangle(box_coordinates, outline="red", width=10)

                            # Save the final image
                            # output_path = 'path/to/your/output/image_with_bbox.jpg'
                            # image_draw.save(path_images + r"_result/" + filename)
                        except:
                            print(f"bboxes {bboxes}")
                            print(f"scores {scores}")
                            print(f"labels {labels}")
                            path_to_print = path_images + r"/" + filename
                            print(f"path {path_to_print}")
                            raise ValueError("Something went wrong")
                    else:
                        bboxes, labels, scores = filter_similar_boxes(bboxes, labels, scores)

                    if print_single_result:
                        print(f"All filtered boxes: {bboxes}")
                        print(f"All filtered labels: {labels}")
                        print(f"All filtered scores: {scores}")
                # print(bboxes)
                # print(final_dict[int(filename.split(".")[0])])
                if filename in final_dict and bboxes.numel() != 0:
                    if len(bboxes) > 0:
                        for idx, bbox in enumerate(bboxes):
                            label = labels[idx]
                            matching_tensor, pidx = find_best_matching_tensor(bbox, torch.tensor(
                                final_dict[filename]["bboxes"]), threshold_overlap)
                            if print_single_result:
                                print(f"matching tensor: {matching_tensor}")
                            # Calculate 120% of tensor2 for each dimension
                            # tensor2_120_percent = [1.2 * dim for sub_tensor in final_dict[filename]["bboxes"] for dim in sub_tensor]
                            max_allowed_area = 1.3 * calculate_area(final_dict[filename]["bboxes"])
                            min_allowed_area = 0.7 * calculate_area(final_dict[filename]["bboxes"])

                            # if matching_tensor is None or not all(sub_tensor1 <= sub_tensor2 for sub_tensor1, sub_tensor2 in zip(bbox, tensor2_120_percent)):
                            if matching_tensor is None or (max_allowed_area < calculate_area(
                                    bbox) and threshold_overlap > 0.001 and min_allowed_area > calculate_area(bbox)):
                                if print_single_result:
                                    if matching_tensor is not None and threshold_overlap > 0.001:
                                        print(f"max allowed area: {max_allowed_area}")
                                        print(f"area of prediction: {calculate_area(bbox)}")

                                    print("not found: bbox(" + str(bbox) + ") in list of bboxes(" + str(
                                        final_dict[filename]["bboxes"]) + ") for filename: " + str(
                                        filename) + " --> false positive")

                                    # draw_bounding_boxes(path_images + r"/" + filename, [bbox, final_dict[filename]["bboxes"]])
                                # print(x_mean, y_mean, bboxes)
                                false_positive += 1
                            else:
                                label_target = torch.tensor(final_dict[filename]["labels"][pidx])

                                if label == label_target:
                                    if print_single_result:
                                        print("found: bbox(" + str(bbox) + ") in list of bboxes(" + str(
                                            final_dict[filename][
                                                "bboxes"]) + ") for filename: " + str(filename) + " for label " + str(
                                            label_target) + " --> true positive")
                                    # print(x_mean, y_mean, bboxes)
                                    true_positive += 1
                                    del final_dict[filename]["bboxes"][idx]
                                else:
                                    if print_single_result:
                                        print("not found: bbox(" + str(bbox) + ") in list of bboxes(" + str(
                                            final_dict[filename][
                                                "bboxes"]) + ") for filename: " + str(filename) + " for label " + str(
                                            label_target) + " --> false positive")
                                    false_positive += 1
                                    # draw_bounding_boxes(path_images + r"/" + filename, [bbox, final_dict[filename]["bboxes"]])
                                    # raise RuntimeError("Debugging")
                        # if len(final_dict[filename]["bboxes"]) > 0:
                        #     for bbox in final_dict[filename]["bboxes"]:
                        #         if print_single_result:
                        #             print("not detected from model: bbox: " + str(bbox) + " and filename: " + str(filename) + " --> false negative")
                        #         false_negative += 1
                    #
                    #
                    #
                    #         try:
                    #             x_start, y_start, x_end, y_end = final_dict[filename]["bboxes"]
                    #         except:
                    #             print("ERROR:")
                    #             print(filename)
                    #
                    #         x_mean = ((x_start + x_end) + x_start) / 2
                    #         y_mean = ((y_end + y_start) + y_start) / 2
                    #
                    #         # print(x_mean)
                    #         # print(bboxes[0][0])
                    #         # print(bboxes[0][2])
                    #
                    #         # print(y_mean)
                    #         # print(bboxes[0][1])
                    #         # print(bboxes[0][3])
                    #         # if caption == "speaker":
                    #         #     check = x_mean >= bboxes[0][0] and x_mean <= bboxes[0][2] and y_mean >= bboxes[0][1] and y_mean <= \
                    #         #             bboxes[0][1] + bboxes[0][3]
                    #         # else:
                    #         #     check = 0
                    #         if scores >= threshold:
                    #             if x_mean >= bboxes[0][0] and x_mean <= bboxes[0][2] and y_mean >= bboxes[0][
                    #                 1] and y_mean <= \
                    #                     bboxes[0][1] + bboxes[0][3]:
                    #                 if print_single_result:
                    #                     print("found: " + filename + " --> true positive")
                    #                 true_positive += 1
                    #             else:
                    #                 if print_single_result:
                    #                     print("not found: " + filename + " --> false positive")
                    #                 # print(x_mean, y_mean, bboxes)
                    #                 false_positive += 1
                    #         else:
                    #             if print_single_result:
                    #                 print("Below threshold: " + filename + " --> false negative")
                    #             false_negative += 1
                    #
                    # else:
                    #     if scores >= threshold:
                    #         if print_single_result:
                    #             print("Above threshold " + filename + " --> false positive")
                    #         false_positive += 1
                    #     else:
                    #         if print_single_result:
                    #             print("Below threshold " + filename + " --> true negative")
                    #         true_negative += 1
                    # counter_all += 1
                    else:
                        if len(final_dict[filename]["bboxes"]) > 0:
                            for bbox in final_dict[filename]["bboxes"]:
                                if print_single_result:
                                    print("not detected from model: bbox: " + str(bbox) + " and filename: " + filename +
                                          " --> false negative")
                                false_negative += 1
                        else:
                            if print_single_result:
                                print("not detected from model: bbox: " + str(
                                    final_dict[filename]["bboxes"]) + " and filename: " + filename +
                                      " --> false negative")
                            false_negative += 1
                else:
                    if print_single_result:
                        print("not detected from model: bbox: " + str(
                            final_dict[filename]["bboxes"]) + " and filename: " + filename +
                              " --> false negative")
                    false_negative += 1
        else:
            print(f"filename {filename} not processed")

    # print("precision: " + str(true_positive / (true_positive + false_positive)))
    # print("recall: " + str(true_positive / (true_positive + false_negative)))
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")
    print(f"True positive: {true_positive}")
    print(f"True negative: {true_negative}")
    return {"precision": true_positive / (true_positive + false_positive),
            "recall": true_positive / (true_positive + false_negative),
            "accuracy": (true_positive + true_negative) / (
                        true_positive + true_negative + false_negative + false_positive)
            }


def evaluate_json_voting(path_json: str, path_images: str, glip_own, caption: str = "spokesman. listener",
                         print_single_result: bool = True, threshold_predictions: float = 0.1,
                         threshold_overlap: float = 0.5, transform_annotation_bboxes_format: bool = True):
    # assert "." not in caption, "Yet only active_speaker-single_and_voting captions are supported"
    # Opening JSON file
    f = open(path_json)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # transform_bboxes_length_to_point(data)
    # Closing file
    f.close()
    # Iterating through the json
    # list

    categories = {}
    for pdict in data["categories"]:
        if pdict["name"].lower() in caption.lower():
            categories[pdict["id"]] = pdict["name"].lower()

    final_dict = {}
    for pdict in data["annotations"]:
        if pdict["category_id"] in categories.keys():
            file_name = next((adict["file_name"].replace(":", "_").replace("_0", "").replace("_1", "").replace("_2",
                                                                                                               "").replace(
                "_3", "").split("-")[1] for adict in data["images"] if adict['id'] == pdict["image_id"]), -1)
            if file_name in final_dict:
                final_dict[file_name]["labels"].append(pdict["category_id"])
                final_dict[file_name]["bboxes"].extend(
                    [[i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                     [pdict["bbox"]]])
                # final_dict [final_dict[file_name], pdict["category_id"], pdict["bbox"]]
            else:
                final_dict[file_name] = {"labels": [pdict["category_id"]], "bboxes": [
                    [i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                    [pdict["bbox"]]]}

    files = {}
    for pdict in data["images"]:
        files[pdict["file_name"].replace(":", "_").split("-")[1]] = pdict["id"]

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    counter_all = 0

    results = {}
    # Get a sorted list of filenames
    sorted_filenames = sorted(os.listdir(path_images))
    print(f"sorted_filenames: {sorted_filenames}")

    # Loop through the sorted filenames with tqdm
    for filename in tqdm(sorted_filenames):
        if filename.endswith("jpg"):
            # if int(filename.split(".")[0].split("-")[-1]) in final_dict:
            if filename.replace(":", "_").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                        "") in files or filename in files:
                print(f"filename {filename} found")
                image = Image.open(path_images + r"/" + filename).convert("RGB")
                image = np.array(image)  # [:, :, [2, 1, 0]]
                # caption = 'spokesman'
                result, top_predictions = glip_own.run_on_web_image(image, caption, threshold_predictions)

                scores = top_predictions.get_field("scores")
                bboxes = top_predictions.bbox
                labels = top_predictions.get_field("labels")
                entities = glip_own.entities

                if print_single_result:
                    print(f"All boxes: {bboxes}")
                    print(f"All labels: {labels}")
                    print(f"All scores: {scores}")

                if caption == "spokesman":
                    bboxes, labels, scores = remove_indices_below_max(bboxes, scores, labels)
                else:
                    bboxes, labels, scores = filter_similar_boxes(bboxes, labels, scores)

                if print_single_result:
                    print(f"All filtered boxes: {bboxes}")
                    print(f"All filtered labels: {labels}")
                    print(f"All filtered scores: {scores}")

                if filename.replace(":", "_").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                            "") in results:
                    results[
                        filename.replace(":", "_").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                                 "")].append(
                        bboxes[0].tolist())
                else:
                    print(f"bbox {bboxes.tolist()} add as list [].")
                    if bboxes is None or [bboxes] is None:
                        raise RuntimeError("Something crazy happened here")
                    results[
                        filename.replace(":", "_").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                                 "")] = bboxes.tolist()
                # print(bboxes)
                # print(final_dict[int(filename.split(".")[0])])

            else:
                print(f"filename {filename} not found in files {files}")

    for k, v in tqdm(results.items()):
        # for k, v in results.items():
        print(f"key to be looked for: {k}")
        print(f"final_dict keys: {final_dict.keys()}")
        if k in final_dict:
            try:
                bboxes = voting_function(v, threshold_overlap)
            except:
                print(f"voting_function failed for key {k} and value: {v}")
            bboxes = [bboxes]
            if len(bboxes) > 0:
                for idx, bbox in enumerate(bboxes):
                    label = labels[idx]
                    matching_tensor, pidx = find_best_matching_tensor(bbox, torch.tensor(final_dict[k]["bboxes"]),
                                                                      threshold_overlap)
                    if print_single_result:
                        print(f"matching tensor: {matching_tensor}")
                    # Calculate 120% of tensor2 for each dimension
                    # tensor2_120_percent = [1.2 * dim for sub_tensor in final_dict[filename]["bboxes"] for dim in sub_tensor]
                    max_allowed_area = 1.3 * calculate_area(final_dict[k]["bboxes"])
                    min_allowed_area = 0.7 * calculate_area(final_dict[k]["bboxes"])

                    # if matching_tensor is None or not all(sub_tensor1 <= sub_tensor2 for sub_tensor1, sub_tensor2 in zip(bbox, tensor2_120_percent)):
                    if matching_tensor is None or (max_allowed_area < calculate_area(
                            bbox) and threshold_overlap > 0.001 and min_allowed_area > calculate_area(bbox)):
                        if print_single_result:
                            if matching_tensor is not None and threshold_overlap > 0.001:
                                print(f"max allowed area: {max_allowed_area}")
                                print(f"area of prediction: {calculate_area(bbox)}")

                            print("not found: bbox(" + str(bbox) + ") in list of bboxes(" + str(
                                final_dict[k]["bboxes"]) + ") for k: " + str(k) + " --> false positive")

                            # draw_bounding_boxes(path_images + r"/" + filename, [bbox, final_dict[filename]["bboxes"]])
                        # print(x_mean, y_mean, bboxes)
                        false_positive += 1
                    else:
                        label_target = torch.tensor(final_dict[k]["labels"][pidx])

                        if label == label_target:
                            if print_single_result:
                                print("found: bbox(" + str(bbox) + ") in list of bboxes(" + str(final_dict[k][
                                                                                                    "bboxes"]) + ") for k: " + str(
                                    k) + " for label " + str(label_target) + " --> true positive")
                            # print(x_mean, y_mean, bboxes)
                            true_positive += 1
                            del final_dict[k]["bboxes"][idx]
                        else:
                            if print_single_result:
                                print("not found: bbox(" + str(bbox) + ") in list of bboxes(" + str(final_dict[k][
                                                                                                        "bboxes"]) + ") for k: " + str(
                                    k) + " for label " + str(label_target) + " --> false positive")
                            false_positive += 1
                            # draw_bounding_boxes(path_images + r"/" + filename, [bbox, final_dict[filename]["bboxes"]])
                            # raise RuntimeError("Debugging")
                # if len(final_dict[filename]["bboxes"]) > 0:
                #     for bbox in final_dict[filename]["bboxes"]:
                #         if print_single_result:
                #             print("not detected from model: bbox: " + str(bbox) + " and filename: " + str(filename) + " --> false negative")
                #         false_negative += 1
            #
            #
            #
            #         try:
            #             x_start, y_start, x_end, y_end = final_dict[filename]["bboxes"]
            #         except:
            #             print("ERROR:")
            #             print(filename)
            #
            #         x_mean = ((x_start + x_end) + x_start) / 2
            #         y_mean = ((y_end + y_start) + y_start) / 2
            #
            #         # print(x_mean)
            #         # print(bboxes[0][0])
            #         # print(bboxes[0][2])
            #
            #         # print(y_mean)
            #         # print(bboxes[0][1])
            #         # print(bboxes[0][3])
            #         # if caption == "speaker":
            #         #     check = x_mean >= bboxes[0][0] and x_mean <= bboxes[0][2] and y_mean >= bboxes[0][1] and y_mean <= \
            #         #             bboxes[0][1] + bboxes[0][3]
            #         # else:
            #         #     check = 0
            #         if scores >= threshold:
            #             if x_mean >= bboxes[0][0] and x_mean <= bboxes[0][2] and y_mean >= bboxes[0][
            #                 1] and y_mean <= \
            #                     bboxes[0][1] + bboxes[0][3]:
            #                 if print_single_result:
            #                     print("found: " + filename + " --> true positive")
            #                 true_positive += 1
            #             else:
            #                 if print_single_result:
            #                     print("not found: " + filename + " --> false positive")
            #                 # print(x_mean, y_mean, bboxes)
            #                 false_positive += 1
            #         else:
            #             if print_single_result:
            #                 print("Below threshold: " + filename + " --> false negative")
            #             false_negative += 1
            #
            # else:
            #     if scores >= threshold:
            #         if print_single_result:
            #             print("Above threshold " + filename + " --> false positive")
            #         false_positive += 1
            #     else:
            #         if print_single_result:
            #             print("Below threshold " + filename + " --> true negative")
            #         true_negative += 1
            # counter_all += 1
            else:
                if len(final_dict[k]["bboxes"]) > 0:
                    for bbox in final_dict[k]["bboxes"]:
                        if print_single_result:
                            print("not detected from model: bbox: " + bbox + " and filename: " + k +
                                  " --> false negative")
                        false_negative += 1

    # print("precision: " + str(true_positive / (true_positive + false_positive)))
    # print("recall: " + str(true_positive / (true_positive + false_negative)))
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")
    print(f"True positive: {true_positive}")
    print(f"True negative: {true_negative}")
    return {"precision": true_positive / (true_positive + false_positive),
            "recall": true_positive / (true_positive + false_negative)}


def calculate_center(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def voting_function(boxes, iou_threshold=0.5):
    n = len(boxes)

    assert n == 3, "Please provide list of bboxes with length 3"

    if n < 2:
        return 0, 0, 0, 0  # Not enough bounding boxes for voting

    max_vote_area = 0
    max_vote_box = (0, 0, 0, 0)

    for i in range(n - 1):
        for j in range(i + 1, n):
            iou, intersection_area, intersection_bbox = bbox_iou(boxes[i], boxes[j], return_intersection_area=True,
                                                                 return_intersection_area_as_bbox=True)

            if iou >= iou_threshold:
                # Consider the area covered by at least 2 overlapping bounding boxes
                vote_area = intersection_area
                # max((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
                # (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]))

                if vote_area > max_vote_area:
                    max_vote_area = vote_area
                    # x_start = min(boxes[i][0], boxes[j][0])
                    # y_start = min(boxes[i][1], boxes[j][1])
                    # x_end = max(boxes[i][2], boxes[j][2])
                    # y_end = max(boxes[i][3], boxes[j][3])
                    max_vote_box = intersection_bbox

    if torch.is_tensor(max_vote_box[0]):
        max_vote_box = list(map(int, max_vote_box))
    return max_vote_box


def draw_bounding_boxes(image_path, bounding_boxes):
    # Read the image
    img = cv2.imread(image_path)

    # Convert bounding box coordinates to integers
    temp_boxes = []
    for element in bounding_boxes:
        if isinstance(element[0], list):
            temp_boxes.append(element[0])
        else:
            temp_boxes.append(element)

    print(temp_boxes)
    bounding_boxes = [bbox.tolist() if isinstance(bbox, np.ndarray) else bbox for bbox in temp_boxes]
    # bounding_boxes = np.array(temp_boxes, dtype=int)
    del temp_boxes

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        x_start, y_start, x_end, y_end = bbox
        cv2.rectangle(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0),
                      2)  # Draw a green rectangle

    # Display the image with bounding boxes
    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to calculate the area of a bounding box
def calculate_area(bbox):
    if bbox is not None:
        if isinstance(bbox[0], list):
            bbox_temp = bbox[0]
        else:
            bbox_temp = bbox

        x_start, y_start, x_end, y_end = bbox_temp
        width = x_end - x_start
        height = y_end - y_start
        area = width * height
        return area
    else:
        raise ValueError("Invalid bbox value has been passed!")


def evaluate_json_distance(path_json: str, path_images: str, glip_own, caption: str = "subtitle-placement",
                           print_single_result: bool = True, threshold_predictions: float = 0.1,
                           transform_annotation_bboxes_format: bool = True, normalize: bool = False):
    # assert "." not in caption, "Yet only active_speaker-single_and_voting captions are supported"
    # Opening JSON file
    f = open(path_json)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # transform_bboxes_length_to_point(data)
    # Closing file
    f.close()
    # Iterating through the json
    # list

    categories = {}
    for pdict in data["categories"]:
        if pdict["name"].lower() in caption.lower():
            categories[pdict["id"]] = pdict["name"].lower()

    final_dict = {}
    for pdict in data["annotations"]:
        if pdict["category_id"] in categories.keys():
            file_name = next((adict["file_name"] for adict in data["images"] if
                              adict['id'] == pdict["image_id"] or int(adict['file_name'].split(".")[0]) == int(
                                  pdict["image_id"])), -1)
            if file_name == -1:
                image_id = pdict["image_id"]
                raise ValueError(f"No image found for image_id {image_id}")
            image_width = next((adict["width"] for adict in data["images"] if adict['id'] == pdict["image_id"]), -1)
            image_height = next((adict["height"] for adict in data["images"] if adict['id'] == pdict["image_id"]), -1)
            if file_name in final_dict:
                print(f"file_name {file_name} already present")
                final_dict[file_name]["labels"].append(pdict["category_id"])
                final_dict[file_name]["bboxes"].extend(
                    [[i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                     [pdict["bbox"]]])
                final_dict[file_name]["width"].append(image_width)
                final_dict[file_name]["height"].append(image_height)
                # final_dict [final_dict[file_name], pdict["category_id"], pdict["bbox"]]
            else:
                final_dict[file_name] = {"labels": [pdict["category_id"]], "bboxes": [
                    [i[0], i[1], i[0] + i[2], i[1] + i[3]] if transform_annotation_bboxes_format else i for i in
                    [pdict["bbox"]]], "height": image_height, "width": image_width}

    files = {}
    for pdict in data["images"]:
        files[pdict["file_name"].replace(":", "_")] = pdict["id"]

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    counter_all = 0

    distances = []
    for filename in tqdm(os.listdir(path_images)):
        if filename.endswith("jpg"):  # and "2445" in filename:
            # if int(filename.split(".")[0].split("-")[-1]) in final_dict:
            if filename.replace(":", "_") in files or filename in files:
                image = Image.open(path_images + r"/" + filename).convert("RGB")
                # image = np.array(image)[:, :, [2, 1, 0]]
                image = np.array(image)
                # caption = 'spokesman'
                result, top_predictions = glip_own.run_on_web_image(image, caption, threshold_predictions)
                scores = top_predictions.get_field("scores")
                bboxes = top_predictions.bbox
                labels = top_predictions.get_field("labels")
                entities = glip_own.entities

                # print(scores)
                # print(labels)
                # print(bboxes)
                # if "2445" in filename:
                #     img = Image.fromarray(result.astype('uint8'))
                #     img.save('/gdrive/MyDrive/DataGlip/OUTPUT_subtitle_middle_middle/output_image.jpg')
                #     exit(-1)
                if len(caption.replace(" ", "").split(".")) == 1:
                    bboxes, labels, scores = remove_indices_below_max(bboxes, scores, labels)
                else:
                    bboxes, labels, scores = filter_similar_boxes(bboxes, labels, scores)
                # print(bboxes)
                # print(final_dict[int(filename.split(".")[0])])
                if filename in final_dict:
                    if len(bboxes) > 0 and len(bboxes) < 2:
                        temp = torch.tensor(final_dict[filename]["bboxes"])
                        distances.append(calculate_distance_between_bounding_boxes(bboxes[0],
                                                                                   torch.tensor(final_dict[filename]
                                                                                                ["bboxes"][0]),
                                                                                   picture_width=torch.tensor(final_dict
                                                                                                              [filename]
                                                                                                              [
                                                                                                                  "width"]),
                                                                                   picture_height=torch.tensor(
                                                                                       final_dict
                                                                                       [filename]
                                                                                       ["height"]),
                                                                                   normalize=normalize))
                    else:
                        results = []
                        for bbox in bboxes:
                            results.append(
                                calculate_distance_between_bounding_boxes(bbox, torch.tensor(final_dict[filename]
                                                                                             ["bboxes"][0]),
                                                                          picture_width=torch.tensor(final_dict
                                                                                                     [filename]
                                                                                                     ["width"]),
                                                                          picture_height=torch.tensor(final_dict
                                                                                                      [filename]
                                                                                                      ["height"]),
                                                                          normalize=normalize))
                        if results == []:
                            results = [torch.sqrt(
                                torch.tensor(final_dict[list(final_dict.keys())[0]]["width"]) ** 2 + torch.tensor(
                                    final_dict[list(final_dict.keys())[0]]
                                    ["height"]) ** 2)]
                        distances.append(mean(results))
                else:
                    print(f"filename {filename} not found in final_dict {final_dict.keys()}")
                    if normalize:
                        distances.append(1)
                    else:
                        distances.append(torch.sqrt(
                            torch.tensor(final_dict[list(final_dict.keys())[0]]["width"]) ** 2 + torch.tensor(
                                final_dict[list(final_dict.keys())[0]]
                                ["height"]) ** 2))

    # print("precision: " + str(true_positive / (true_positive + false_positive)))
    # print("recall: " + str(true_positive / (true_positive + false_negative)))
    return {"distances": sum(distances) / len(distances)}


def calculate_distance_between_bounding_boxes(box1, box2, picture_width, picture_height, normalize: bool = True):
    """
    Calculate the Euclidean distance between the middle points of two bounding boxes and scale it in relation to the picture size.

    Args:
        box1 (torch.Tensor): A tensor representing the first bounding box in the format [x_start, y_start, x_end, y_end].
        box2 (torch.Tensor): A tensor representing the second bounding box in the format [x_start, y_start, x_end, y_end].
        picture_width (int): The width of the picture.
        picture_height (int): The height of the picture.
        normalize: define if the distance is set in relation to the maximal possible distance

    Returns:
        float: The scaled Euclidean distance between the middle points of the two bounding boxes.

    """
    if isinstance(box1, list):
        box1 = torch.tensor(box1)
    if isinstance(box2, list):
        box2 = torch.tensor(box2)
    if isinstance(picture_width, (int, float)):
        picture_width = torch.tensor(picture_width)
    if isinstance(picture_height, (int, float)):
        picture_height = torch.tensor(picture_height)
    # Extract coordinates from the tensors
    x1_start, y1_start, x1_end, y1_end = box1
    x2_start, y2_start, x2_end, y2_end = box2

    # Calculate the middle points of the bounding boxes
    x1_middle = (x1_start + x1_end) / 2
    y1_middle = (y1_start + y1_end) / 2
    x2_middle = (x2_start + x2_end) / 2
    y2_middle = (y2_start + y2_end) / 2

    # Calculate the Euclidean distance between the middle points
    distance = torch.sqrt((x1_middle - x2_middle) ** 2 + (y1_middle - y2_middle) ** 2)

    # Calculate the relative distance
    # relative_distance = distance / ((picture_width + picture_height) / 2)

    max_distance = torch.sqrt(picture_width ** 2 + picture_height ** 2)

    # Normalize the distance
    if normalize:
        return distance / max_distance
    else:
        return distance


def find_equal_objects(tensor):
    """
    Finds objects in `tensor` that are equal.

    Parameters:
        tensor (torch.Tensor): The PyTorch tensor to search for equal objects.

    Returns:
        list: A list of sets, where each set contains the indices of the objects
        that are equal.
    """
    unique_objects, object_counts = tensor.unique(return_counts=True, dim=0)
    equal_objects = [list(set(torch.where(torch.all(tensor == obj, dim=1))[0])) for obj, count in
                     zip(unique_objects, object_counts) if count > 1]
    return equal_objects


def remove_indices_below_max(tensor, reference_tensor, reference_label_tensor):
    """
    Removes the indices of `tensor` that are below the maximum value of
    `reference_tensor`.

    Parameters:
        tensor (torch.Tensor): The PyTorch tensor to remove indices from.
        reference_tensor (torch.Tensor): The PyTorch tensor to use as a
            reference for the maximum value.

    Returns:
        torch.Tensor: The resulting PyTorch tensor with the indices below the
        maximum value of `reference_tensor` removed.
    """
    max_value = torch.max(reference_tensor)
    indices_to_remove = torch.where(reference_tensor < max_value)[0]
    tensor = torch.index_select(tensor, 0, torch.tensor([i for i in range(len(tensor)) if i not in indices_to_remove]))
    reference_label_tensor = torch.index_select(reference_label_tensor, 0, torch.tensor(
        [i for i in range(len(reference_label_tensor)) if i not in indices_to_remove]))
    return tensor, reference_label_tensor, max_value


def main_batch(config_files, weight_files, json_files, path_images, task, prompt, normalize,
               threshold_overlap: float = 0.5, print_results: bool = False, position_height=80):
    for i in range(0, len(config_files)):
        glip_demo = initialize(config_file=config_files[i], weight_file=weight_files[i])
        glip_demo.color = 175
        if task == "recall":
            result_dict = evaluate_json(path_json=json_files[i], path_images=path_images[i], glip_own=glip_demo,
                                        caption=prompt, print_single_result=print_results,
                                        threshold_overlap=threshold_overlap)
        elif task == "recall_voting":
            result_dict = evaluate_json_voting(path_json=json_files[i], path_images=path_images[i], glip_own=glip_demo,
                                               caption=prompt, print_single_result=print_results,
                                               threshold_overlap=threshold_overlap)
        elif task == "distance":
            result_dict = evaluate_json_distance(path_json=json_files[i], path_images=path_images[i],
                                                 glip_own=glip_demo,
                                                 caption=prompt, print_single_result=print_results, normalize=normalize)
        elif task == "distance_subtile_placement":
            result_dict = evaluate_subtitles_with_json_target(path_json=json_files[i], path_images=path_images[i],
                                                              glip_own=glip_demo, caption=prompt,
                                                              print_single_result=False,
                                                              normalize=normalize, position_height=position_height)
        else:
            raise ValueError("Provided task " + task + " is not available.")

        print(result_dict)
        # result_dict = evaluate_json(path_json=json_files[i], path_images=path_images[i], glip_own=glip_demo,
        #                             caption="listener", print_single_result=False)
        #
        # print(result_dict)
    # image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')
    # caption = 'bobble heads on top of the shelf'
    # result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
    # imshow(result, caption)


def transform_bboxes_length_to_point(annotation: dict):
    # print(f"annotation: {annotation}")
    # print(annotation["annotations"])
    for idx, element in enumerate(annotation["annotations"]):
        if isinstance(element["bbox"][0], list):
            for pidx, bbox in enumerate(annotation["annotations"][idx]["bbox"]):
                print(f"before: {bbox}")
                annotation["annotations"][idx]["bbox"][pidx] = [bbox["bbox"][0], bbox["bbox"][1],
                                                                bbox["bbox"][0] + bbox["bbox"][2],
                                                                bbox["bbox"][1] + bbox["bbox"][3]]
                print(f"after: {annotation['annotations'][idx]['bbox'][pidx]}")
        else:
            print(f"before: {element}")
            annotation["annotations"][idx]["bbox"] = [element["bbox"][0], element["bbox"][1],
                                                      element["bbox"][0] + element["bbox"][2],
                                                      element["bbox"][1] + element["bbox"][3]]
            # [[x_start, y_start, x_start + x_length, y_start + y_length] for x_start, y_start, x_length, y_length in annotation["annotations"][idx]["bbox"]]
            print(f"after: {annotation['annotations'][idx]['bbox']}")
        # raise ValueError("Just debugging")


def main_single(filename: str, config_file: str, weight_file: str, path_images: str, prompt: str = "subtitle-position",
                filter_for_max: bool = True):
    # config_file = r"E:\Python\MasterThesis\GLIP\model\normal\config.yml"
    # weight_file = r"E:\Python\MasterThesis\GLIP\model\normal\model_best.pth"

    # path_images = r"E:\Python\MasterThesis\GLIP\DATASET\ASD\images"

    glip_demo = initialize(config_file=config_file, weight_file=weight_file)
    glip_demo.color = 175
    image = Image.open(path_images + r"/" + filename).convert("RGB")
    # image_array = np.array(image)[:, :, [2, 1, 0]]
    image_array = np.array(image)

    result, top_predictions = glip_demo.run_on_web_image(image_array, prompt, 0.1)
    output_path = '/gdrive/MyDrive/DataGlip/OUTPUT_subtitle_middle_empty/' + filename
    if filter_for_max:
        scores = top_predictions.get_field("scores")
        bboxes = top_predictions.bbox
        print(f"bboxes before filtering: {bboxes}")
        labels = top_predictions.get_field("labels")
        bboxes, labels, scores = remove_indices_below_max(bboxes, scores, labels)
        print(f"bboxes after filtering: {bboxes}")

        # Draw bounding box on the image
        draw = ImageDraw.Draw(image)
        bbox_to_imprint = tuple(bboxes[0].cpu().numpy())  # Assuming there is at least one bounding box
        print(f"bbox_to_imprint: {bbox_to_imprint}")
        draw.rectangle(bbox_to_imprint, outline="red", width=2)
    else:
        image = Image.fromarray(result.astype('uint8'))

    image.save(output_path)

    # glip_demo = initialize(config_file=config_file, weight_file=weight_file)
    # glip_demo.color = 175
    # image = Image.open(path_images + r"/" + filename).convert("RGB")
    # image = np.array(image)[:, :, [2, 1, 0]]

    # result, top_predictions = glip_demo.run_on_web_image(image, prompt, 0.1)
    # bboxes, labels, scores = remove_indices_below_max(bboxes, scores, labels)
    # imsave(image, '/gdrive/MyDrive/DataGlip/OUTPUT_subtitle_middle_empty/' + filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of COCO annotated results")

    parser.add_argument("--config", help="Path to config file", type=str, nargs='+')
    parser.add_argument("--weight", help="Path to weight/checkpoint file", type=str, nargs='+')
    parser.add_argument("--annotation", help="Path to annotation json file", type=str, nargs='+')
    parser.add_argument("--img", help="Path to images", type=str, nargs='+')
    parser.add_argument("--task",
                        help="Define task: distance or recall or to show predicted bounding box for a active_speaker-single_and_voting image, chhose active_speaker-single_and_voting",
                        type=str)
    parser.add_argument("--prompt", help="Define prompt for execution/evaluaiton.", type=str)
    parser.add_argument("--normalize", action="store_true", help="Define if normalizing should be applied or not")
    parser.add_argument("--print_result", action="store_true",
                        help="Define if active_speaker-single_and_voting evaluation results should be displayed.")
    parser.add_argument("--position_height", type=int,
                        help="Defines the positioning height for subtitle placement in % of the height of image.",
                        default=80)
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Define the iou threshold for task recall to add restriction for matchiing predicted and actual bounding box.")
    parser.add_argument("--filter_for_max", action="store_true",
                        help="Defines if for the active_speaker-single_and_voting task all found bboxes or only max bbox should be imprinted.")
    args = parser.parse_args()

    config_files = args.config
    weight_files = args.weight
    json_files = args.annotation
    path_images = args.img
    task = args.task.lower()
    prompt = args.prompt
    normalize = args.normalize
    threshold_overlap = args.iou_threshold
    print_result = args.print_result
    position_height = args.position_height
    filter_for_max = args.filter_for_max

    print(f"--filter_for_max: {filter_for_max}")

    # config_files = [r"D:\Gits\GLIP\model\SBPLM\config.yml"]
    # weight_files = [r"D:\Gits\GLIP\model\SBPLM\model_best.pth"]
    # json_files = [r"D:\Gits\GLIP\DATASET\SBPLM\stuff_val2017_GLIP.json"]
    # path_images = [r"E:\Python\MasterThesis\AutomaticSubtitlePlacement\data\subtitle_position_boxes_middle_of_subtitle\_A\val"]
    if task.lower() != "active_speaker-single_and_voting":
        if "distance" in task.lower():
            print(f"Normalize distances: {normalize}")
        main_batch(config_files, weight_files, json_files, path_images, task, prompt, normalize,
                   print_results=print_result, threshold_overlap=threshold_overlap, position_height=position_height)
    else:
        for idx, ppath in enumerate(path_images):
            main_single(ppath.split(r"/")[-1], config_files[idx], weight_files[idx], r"/".join(ppath.split(r"/")[:-1]),
                        prompt, filter_for_max=filter_for_max)

    # main_single(filename=r"2122.jpg", config_file=config_files[0], weight_file=weight_files[0], path_images=path_images[0])

    # config_files = [r"E:\Python\MasterThesis\GLIP\model\Overlapped\config.yml",
    #                r"E:\Python\MasterThesis\GLIP\model\normal\config.yml"]
    # weight_files = [r"E:\Python\MasterThesis\GLIP\model\Overlapped\model_best.pth",
    #                r"E:\Python\MasterThesis\GLIP\model\normal\model_best.pth"]
    # json_files = [r"E:\Python\MasterThesis\GLIP\DATASET\active_speaker-overlapped\result_val.json",
    #              r"E:\Python\MasterThesis\GLIP\DATASET\normal\result_val.json"]
    # path_images = [r"E:\Python\MasterThesis\GLIP\DATASET\active_speaker-overlapped\images",
    #               r"E:\Python\MasterThesis\GLIP\DATASET\normal\images"]

    # main_batch(config_files, weight_files, json_files, path_images)