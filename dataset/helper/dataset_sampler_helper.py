import json
import os
import pathlib
import warnings
import statistics

# import leb128
# import numpy as np
# from typing import List
from datetime import datetime
import random
from segment_anything import sam_model_registry, SamPredictor

from PIL import Image, ImageDraw
from pycocotools.mask import decode, encode, frPyObjects, area
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


# import sys

def __init_arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provide input parameters")

    parser.add_argument("--crop", dest="crop_video", action="store_true", default=False,
                        help='define if black borders should be removed from Video (longer runtime)')

    parser.add_argument("--vo", dest="video_offset", default=None, action="store", type=float,
                        help='define video offset hardcoded. Please provide either --vo or --vd')

    parser.add_argument("--vd", dest="video_duration", default=None, action="store", type=float,
                        help='define video duration hardcoded. Please provide either --vo or --vd')

    parser.add_argument("--vp", dest="video_path", default=str(pathlib.Path().resolve().
                                                               joinpath("video",
                                                                        "video.mp4"
                                                                        )),
                        action="store", type=str,
                        help='define offset hardcoded. Please provide either --vo or --vd')

    parser.add_argument("--bs", dest="batch_size", default=1, action="store", type=int,
                        help='define batch size')

    parser.add_argument("--rf", dest="return_frames", default=False, action="store_true",
                        help='defines if frames with subtitles or frames with separate target position vector is '
                             'returned')

    parser.add_argument("--nf", dest="n_frames", default=200, action="store", type=int,
                        help='define number of frames')

    parser.add_argument("--box", dest="output_box", default=False, action="store_true",
                        help='defines if frames with boxes (black) for subtitle position are displayed instead of '
                             'subtitles')

    parser.add_argument("--td", dest="train_data", default=False, action="store_true",
                        help='defines if training or test dataset should be created')

    parser.add_argument("--tv", dest="validation_data", default=False, action="store_true",
                        help='defines if training or test dataset should be created')

    parser.add_argument("--rc", dest="return_captions", default=False, action="store_true",
                        help='defines if captions json file should be provided')

    parser.add_argument("--riwa", dest="remove_images_without_annotations", default=False,
                        action="store_true", help='defines if images in json files should only contain image '
                                                  'references having any annotation (or caption)')

    parser.add_argument("--stp", dest="subtitles_path", default=str(pathlib.Path().resolve().
                                                                    joinpath("subtitles",
                                                                             "subtitles.xml")),
                        action="store", type=str, help="Provide path to subtitles")

    parser.add_argument("--frc", dest="fractions", default=[0.75, 0.15, 0.1], type=float, nargs="+",
                        help="List of values to define fractions of train, val and test")

    parser.add_argument("--tsk", dest="task", default="dataset_generation_1", type=str, action="store")

    parser.add_argument("--tkn", dest="task_name", type=str, default="subtitle_position_boxes",
                        action="store", help="define task name to store data under")

    parser.add_argument("--emd", dest="extract_middle_and_default", action="store_true", default=False,
                        help="Defines if only positioned (according to subtitle file) and middle positioned framed "
                             "will be returned or another directory will be provided, that contains uninpainted frame "
                             "and positioned frame as well. Can only be True if extract_middle_and_default is True, "
                             "too.")

    parser.add_argument("--frs", dest="frame_step", type=int, action="store", default=6)

    parser.add_argument("--fps", dest="frames_per_step", type=int, action="store", default=1)

    parser.add_argument("--mos", dest="dot_middle_of_subtitle_box", action="store_true", default=False)

    parser.add_argument("--fsp", dest="fixed_start_point", type=int, action="store", default=100)

    parser.add_argument("--ean", dest="extract_annotations", action="store_true", default=True)

    parser.add_argument("--igd", dest="ignore_different", action="store_true", default=False)

    parser.add_argument("--overlay_frames", dest="overlay_frames", type=int, default=1)

    parser.add_argument("--overlay_frames_skip", dest="overlay_frames_skip", type=int, default=1)

    parser.add_argument("--anno", dest="annotations",
                        default=[
                            r"D:\Master_Thesis_data\Active_Speaker\dataset_processed\train"
                            r"\result.json",
                            r"D:\Master_Thesis_data\Active_Speaker\dataset_processed\val\result.json",
                            r"D:\Master_Thesis_data\Active_Speaker\dataset_processed\test\result.json"
                        ],
                        type=str, nargs="+", help="List of annotations paths (train, val, test)")
    return parser.parse_args()


def separate_audio_from_video(path: str = r"C:\Users\Fabia\Videos\Captures\Test Cyrill.mp4",
                              path_target: str = r"C:\Users\Fabia\Videos\Captures\Test_Cyrill.wav"):
    assert path != "", "path for separate_audio_from_video is empty"
    assert path_target != "", "path_target for separate_audio_from_video is empty"
    import moviepy.editor as mp

    my_clip = mp.VideoFileClip(path)
    my_clip.audio.write_audiofile(path_target)


def speech_recognition(path: str) -> str:
    assert path != "", "Path for speech_recognition is empty"
    import speech_recognition as sr

    r = sr.Recognizer()
    with sr.AudioFile(path_target) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        return r.recognize_google(audio_data)


def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def round_base(x, base=200):
    return base * round(x / base)


def display_output_video_subtitled(parsed_args, subtitles_array: np.ndarray):
    import cv2
    # reading the input

    assert parsed_args.video_path is not None and parsed_args.video_path != "", "Please provide proper video path"
    assert parsed_args.video_offset is not None or parsed_args.video_duration is not None, 'Please provide either ' \
                                                                                           '--vo or --vd command ' \
                                                                                           'line parameter'
    cap = cv2.VideoCapture(parsed_args.video_path)
    assert cap.isOpened() is True, "Error reading video file"

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if parsed_args.video_offset is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        offset = duration - parsed_args.video_duration
    else:
        offset = parsed_args.video_offset
    if parsed_args.crop_video is False:
        output = cv2.VideoWriter(
            "output.avi", cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (frame_width, frame_height))
    else:
        output = None

    # output = cv2.VideoWriter(
    #    "output.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
    #    25, (frame_width, frame_height))
    # 30, (1080, 1920))

    i = 0
    while (True):
        # while (i < 2000):
        ret, frame = cap.read()
        i = i + 1
        if (ret):

            # adding filled rectangle on each frame
            if parsed_args.crop_video:
                frame = crop(frame)
                if round_base(frame.shape[0]) != frame_height or round_base(frame.shape[1]) != frame_width:
                    frame_height = round_base(frame.shape[0])
                    frame_width = round_base(frame.shape[1])
                if output is None:
                    output = cv2.VideoWriter(
                        "output.avi", cv2.VideoWriter_fourcc(*'MJPG'),
                        25, (frame_width, frame_height))

            time_frame = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2) + offset
            # print()
            # print(subtitles_array[(subtitles_array[:,1].astype(float) >= time_frame) & (subtitles_array[:,0].astype(float) <= time_frame)].shape)
            # print(subtitles_array[(subtitles_array[:,1].astype(float) >= time_frame) & (subtitles_array[:,0].astype(float) <= time_frame)])
            for begin, end, origin_x, origin_y, extent_x, extent_y, text, region in \
                    subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                                    (subtitles_array[:, 0].astype(float) <= time_frame)]:
                # cv2.rectangle(frame, (100, 150), (500, 600),
                #              (0, 255, 0), -1)
                #    print(begin)
                #    print(time_frame)
                #    print(end)
                cv2.putText(frame, text, (int(frame_width * (float(origin_x.strip("%")) / 100)),
                                          int(frame_height * (float(origin_y.strip("%")) / 100))),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.25, (0, 255, 0))

            # writing the new frame in output
            # print("Frame: {:d}".format(i))
            # print("Ttimestamp is: ", str(time_frame))
            if parsed_args.video_subtitle_output:
                output.write(frame)

            if parsed_args.video_subtitle_display:
                cv2.imshow("output", frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    output.release()
    cap.release()
    cv2.destroyAllWindows()


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
    # import tensorflow as tf
    # frame = tf.image.convert_image_dtype(frame, tf.float32)
    # frame = tf.image.resize_with_pad(frame, *output_size)
    # return frame


class final_sampling:

    def run(self):
        self.filter_dataset(path_dataset=self.path_dataset, rebalance=self.rebalance)
        raise NotImplementedError("Please extend for copying files in final directories")

    def process_files(self, iteration_list):
        root, dirs, files = iteration_list

        copy_dict = {}
        image_height = -100
        image_width = -100

        if root.split("\\")[-2] == "A" or root.split("\\")[-2] == "B" or root.split("\\")[-2] == "_A":
            for file in files:
                if ".png" in file or ".jpg" in file or ".jpeg" in file and image_width == -100 or image_height == -100:
                    image_width, image_height = final_sampling.get_image_dimensions(os.path.join(root, file))

                if root.split("\\")[-2].upper() == "A":
                    if "-".join(file.split("-")[:1]) in self.data_dict[
                        [folder_name for folder_name in os.listdir(self.path_dataset) if
                         folder_name.endswith("single")][0]][os.path.basename(root)]:
                        copy_dict[os.path.join(root, file)] = [
                            "\\subtitle_placement\\single_default\\Pix2Pix\\A\\" + os.path.basename(
                                root) + "\\" + file,
                            "\\subtitle_placement\\single_default\\SAN\\A\\" + os.path.basename(
                                root) + "\\" + file, [(1024, 1024),
                                                      "\\subtitle_placement\\single_default\\DALL-E\\" + os.path.basename(
                                                          root) + "\\" + file]]

                if root.split("\\")[-2].upper() == "_A":
                    if "-".join(file.split("-")[:1]) in self.data_dict[
                        [folder_name for folder_name in os.listdir(self.path_dataset) if "single" in folder_name][
                            0]][os.path.basename(root)]:
                        copy_dict[os.path.join(root, file)] = [
                            "\\subtitle_placement\\single_empty\\Pix2Pix\\A\\" + os.path.basename(
                                root) + "\\" + file,
                            "\\subtitle_placement\\single_empty\\SAN\\A\\" + os.path.basename(
                                root) + "\\" + file, [(1024, 1024),
                                                      "\\subtitle_placement\\single_empty\\DALL-E\\A\\" + os.path.basename(
                                                          root) + "\\" + file]]

                elif root.split("\\")[-2].upper() == "B":
                    if "-".join(file.split("-")[:1]) in self.data_dict[
                        [folder_name for folder_name in os.listdir(self.path_dataset) if "single" in folder_name][
                            0]][os.path.basename(root)]:
                        copy_dict[os.path.join(root, file)] = [
                            "\\subtitle_placement\\single_default\\Pix2Pix\\B\\" + os.path.basename(
                                root) + "\\" + file,
                            "\\subtitle_placement\\single_empty\\Pix2Pix\\B\\" + os.path.basename(
                                root) + "\\" + file]

                elif root.split("\\")[-2].upper() == "PIXELMAPS":
                    if "-".join(file.split("-")[:1]) in self.data_dict[
                        [folder_name for folder_name in os.listdir(self.path_dataset) if "single" in folder_name][
                            0]][os.path.basename(root)]:
                        copy_dict[os.path.join(root, file)] = [
                            "\\subtitle_placement\\single_default\\SAN\\pixelmaps\\" + os.path.basename(
                                root) + "\\" + file,
                            "\\subtitle_placement\\single_empty\\SAN\\pixelmaps\\" + os.path.basename(
                                root) + "\\" + file]

                elif root.split("\\")[-2].upper() == "JSONS":
                    if "-".join(file.split("-")[:1]) in self.data_dict[
                        [folder_name for folder_name in os.listdir(self.path_dataset) if "single" in folder_name][
                            0]][os.path.basename(root)]:
                        copy_dict[os.path.join(root, file)] = [
                            "\\subtitle_placement\\single_default\\SAN\\pixelmaps\\" + os.path.basename(
                                root) + "\\" + file,
                            "\\subtitle_placement\\single_empty\\SAN\\pixelmaps\\" + os.path.basename(
                                root) + "\\" + file]

            if image_width > 0 and image_height > 0:
                generated_image = final_sampling.generate_dall_e_mask(image_height=image_height, image_width=image_width)
                if "to_create" in copy_dict:
                    copy_dict["to_create"].append(
                        [generated_image, "\\subtitle_placement\\single_empty\\DALL-E\\mask.png"])
                else:
                    copy_dict["to_create"] = [
                        [generated_image, "\\subtitle_placement\\single_empty\\DALL-E\\mask.png"]]
                generated_image = final_sampling.generate_dall_e_mask(size=0, image_height=image_height, image_width=image_width)
                if "to_create" in copy_dict:
                    copy_dict["to_create"].append(
                        [generated_image, "\\subtitle_placement\\single_default\\DALL-E\\mask.png"])
                else:
                    copy_dict["to_create"] = [[generated_image,
                                               "\\subtitle_placement\\single_default\\DALL-E\\mask.png"]]  # if dirs in data_dict:  #     # Create folder in output_path if it doesn't exist  #     output_folder = os.path.join(output_path, folder_name)  #     if not os.path.exists(output_folder):  #         os.makedirs(output_folder)  #  #     # Iterate over files in the subfolder  #     for file_name in os.listdir(os.path.join(path_dataset, folder_name)):  #         # Perform filtering based on data in data_dict  #         filtered_data = [data for data in data_dict[folder_name] if data[  #             'filter_criteria'] == file_name]  # Modify the filter_criteria as per your data  #         # Write filtered data to output_path  #         with open(os.path.join(output_folder, file_name), 'w') as output_file:  #             json.dump(filtered_data, output_file)

        return copy_dict

    @staticmethod
    def generate_dall_e_mask(size: int = 30, image_height: int = 1024, image_width: int = 1024):
        from PIL import Image, ImageDraw
        # Create a new image with the desired size and white background
        image_size = (image_width, image_height)
        mask_image = Image.new("RGBA", image_size, color=(255, 255, 255, 255))

        # Create a mask with the same size
        mask = Image.new("L", image_size, 0)

        # Draw a shape on the mask to define the area you want to make transparent
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(round((image_width / 2) - (size / 2)), round(image_height * 0.9 - (size / 2))),
                        (round((image_width / 2) + (size / 2)), round(image_height * 0.9 + (size / 2)))],
                       fill=255)  # Adjust coordinates as needed

        # Apply the mask to the image
        mask_image.putalpha(mask)

        return mask_image


    @staticmethod
    def generate_dall_e_mask(size: int = 30, image_height: int = 1024, image_width: int = 1024):
        # Create a new image with the desired size and white background
        image_size = (image_width, image_height)
        mask_image = Image.new("RGBA", image_size, color=(255, 255, 255, 255))

        # Create a mask with the same size
        mask = Image.new("L", image_size, 0)

        # Draw a shape on the mask to define the area you want to make transparent
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(round((image_width / 2) - (size / 2)), round(image_height * 0.9 - (size / 2))),
                        (round((image_width / 2) + (size / 2)), round(image_height * 0.9 + (size / 2)))],
                       fill=255)  # Adjust coordinates as needed

        # Apply the mask to the image
        mask_image.putalpha(mask)

        return mask_image


    @staticmethod
    def get_image_dimensions(image_path):
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return width, height
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print("An error occurred:", e)


    def read_in_json_files(self, path_jsons: str):
        for root, dirs, files in os.walk(path_jsons):
            base_folder = root.replace(path_jsons, "").replace(os.path.basename(root), "").strip("\\")
            for file in files:
                if file.endswith('.json'):
                    folder_name = os.path.basename(root)
                    with open(os.path.join(root, file), 'r') as json_file:
                        data = json.load(json_file)
                        data = ["-".join(el["file_name"].split("-")[1:]) for el in data["images"]]
                        if base_folder not in self.data_dict:
                            self.data_dict[base_folder] = {}

                        if folder_name not in self.data_dict[base_folder]:
                            self.data_dict[base_folder][folder_name] = []
                        self.data_dict[base_folder][folder_name].extend(data)


    def __init__(self, rebalance, video_path, path_dataset: str =r'G:\Coding\VLM_subtitling\dataset_processed', path_jsons: str = r'G:\Coding\VLM_subtitling\dataset_labeled', ignore_different: bool = True):

        self.rebalance = rebalance
        self.data_dict = {}
        self.read_in_json_files(path_jsons)
        self.path_dataset = path_dataset
        self.path_jsons = path_jsons
        self.final_result_dict = {}
        self.video_path = video_path
        # self.args = args
        self.output_path = "\\".join(video_path.split("\\")[:-1]) + r"\dataset_final"
        self.ignore_different = ignore_different

        # sampler = final_sampling(rebalance)
        self.rebalance_annotated_datasaet(self.data_dict, ignore_different=self.ignore_different)

    def rebalance_annotated_datasaet(self, data_dict, ignore_different):
        # data_dict_temp = copy.deepcopy(data_dict)
        test = 0
        train = 0
        val = 0
        sum_instances = - 1
        for pkey, pvalue in data_dict.items():
            final_files_list = []
            try:
                if "test" in data_dict[pkey]:
                    test = len(data_dict[pkey]["test"])
                    final_files_list.extend(data_dict[pkey]["test"])
                else:
                    test = 0
                if "train" in data_dict[pkey]:
                    train = len(data_dict[pkey]["train"])
                    final_files_list.extend(data_dict[pkey]["train"])
                else:
                    train = 0
                if "val" in data_dict[pkey]:
                    val = len(data_dict[pkey]["val"])
                    final_files_list.extend(data_dict[pkey]["val"])
                else:
                    val = 0
            except:
                raise ValueError("Unexpected data structure found")

            if sum_instances > 0:
                if ignore_different is False:
                    assert sum_instances == test + train + val, (f"You have sorted out different frames in the "
                                                                 f"respective datasets during manual annotation. "
                                                                 f"This results in different data records. "
                                                                 f"Please set the parameter ignore_different to "
                                                                 f"True to suppress this error and to sample based "
                                                                 f"on active speaker detection for single "
                                                                 f"experiment. Key {pkey} differs "
                                                                 f"in {abs(sum_instances - (test + train + val))}")

            sum_instances = test + train + val
            if train > 0:
                train_rebalance = int(sum_instances * self.rebalance[0])
            else:
                train_rebalance = 0
            if val > 0:
                val_rebalance = int(sum_instances * self.rebalance[1])
            else:
                val_rebalance = 0

            if sum_instances == test:
                test_rebalance = test
            else:
                test_rebalance = int(sum_instances * self.rebalance[2])

            assert train_rebalance + val_rebalance + test_rebalance <= sum_instances, (
                "Logic issue, please fix "
                "code.")

            train_images = sorted(final_files_list[:train_rebalance + val_rebalance + test_rebalance])[
                           :train_rebalance]
            val_images = sorted(final_files_list[:train_rebalance + val_rebalance + test_rebalance])[
                         train_rebalance:train_rebalance + val_rebalance]
            test_images = sorted(final_files_list[:train_rebalance + val_rebalance + test_rebalance])[
                          train_rebalance + val_rebalance:]

            data_dict[pkey]["test"] = test_images
            data_dict[pkey]["val"] = val_images
            data_dict[pkey]["train"] = train_images


    def filter_dataset(self, path_dataset, rebalance: list = [0.75, 0.15, 0.1]):
        from PIL import Image, ImageDraw

        import os
        import json  # , copy
        from multiprocessing import Pool
        assert len(rebalance) == 3, "The list should have exactly three values."
        assert sum(rebalance) == 1, "The sum of the values in the list should equal 1."

        # Create a dictionary to store data based on folder names

        # Iterate over subfolders in path_dataset
        assert any("single" in folder_name for folder_name in os.listdir(path_dataset)) and any(
            "overlapped" in folder_name for folder_name in os.listdir(path_dataset)) and any(
            "voting" in folder_name for folder_name in os.listdir(path_dataset)), (
            "Please follow the instructions in the README.md file. "
            "Provide three separate folders for single, overlapping, and voting experiment. Ensure it is the "
            "correct path")

        root_dirs = os.walk(os.path.join(path_dataset, [folder_name for folder_name in os.listdir(path_dataset) if
                                                        "single" in folder_name][0]))


        # for root, dirs, files in root_dirs:
        #    print(self.process_files([root, dirs, files]))

        with Pool() as pool:
            results = pool.map(self.process_files, root_dirs)

        # Now you can collect and aggregate the results

        for result in results:
            self.final_result_dict.update(result)



class SubtitlePlacement:
    @staticmethod
    def _check_args_init(args: argparse.Namespace) -> None:
        """

        :rtype: None
        """
        assert isinstance(args.video_path, str) and len(args.video_path) > 0, "Please provide valid video_path"
        assert isinstance(args.subtitles_path, str) and len(args.subtitles_path) > 0, ("Please provide "
                                                                                       "valid subtitles_path")
        assert isinstance(args.task_name, str) and len(args.task_name) > 0, "Please provide valid task_name"
        assert ((isinstance(args.video_offset, float) and args.video_offset >= 0) or
                (args.video_offset is None and isinstance(args.video_duration, float) and args.video_duration > 0)), \
            "Please provide valid video_duration or video_offset"
        assert isinstance(args.n_frames, int) and args.n_frames > 0, "Please provide valid n_frames"
        assert ((isinstance(args.video_duration, float) and args.video_duration >= 0) or
                (args.video_duration is None and isinstance(args.video_offset, float) and args.video_offset > 0)), \
            "Please provide valid video_duration or video_offset"
        assert isinstance(args.frame_step, int) and args.frame_step > 0, "Please provide valid frame_step"
        assert isinstance(args.fractions, list) and len(args.fractions) == 3, "Please provide valid fractions"
        assert isinstance(args.extract_middle_and_default, bool), "Please provide valid extract_middle_and_default"
        assert isinstance(args.dot_middle_of_subtitle_box, bool), "Please provide valid dot_middle_of_subtitle_box"
        assert isinstance(args.fixed_start_point, int) and args.fixed_start_point >= 0, ("Please provide valid "
                                                                                         "fixed_start_point")
        assert isinstance(args.extract_annotations, bool), "Please provide valid extract_annotations"
        assert isinstance(args.overlay_frames_skip, int) and args.overlay_frames_skip >= 0, ("Please provide valid "
                                                                                             "overlay_frames_skip")
        assert isinstance(args.frames_per_step, int) and args.frames_per_step >= 0, ("Please provide valid "
                                                                                     "frames_per_step")

    def __init_annotation_extract(self):
        self.annotations = {"train": [], "val": [], "test": []}
        self.annotation_info = {
            "description": "SWAT dataset in COCO format",
            "url": "N/A",
            "version": "1.0",
            "year": 2023,
            "contributor": "F.Kneer",
            "date_created": str(datetime.now())
        }
        self.annotation_licences = [
            {
                "url": "N/A",
                "id": 1,
                "name": "SWAT dataset in COCO format"
            }
        ]
        # self.annotation_categories = [
        #     {
        #         "supercategory": "person",
        #         "id": 1,
        #         "name": "spokesman"
        #     },
        #     {
        #         "supercategory": "person",
        #         "id": 2,
        #         "name": "listener"
        #     },
        #     {
        #         "supercategory": "subtitle",
        #         "id": 3,
        #         "name": "subtitle-position"
        #     }
        # ]
        self.annotation_categories = [
            {
                "supercategory": "subtitle",
                "id": 200,
                "name": "subtitle-position"
            }
        ]
        self.image_annotation_info = {"train": [], "val": [], "test": []}

    def __init__(self, video_path: str, subtitles_path: str, video_offset: float, **kwargs):
        # ToDo: add asserts for kwargs
        self.initialized = False
        assert isinstance(subtitles_path, str) and subtitles_path != "", "Please provide valid path for subtitles"
        assert video_path is not None and video_path != "", "Please provide proper video path"
        assert (isinstance(video_offset, (float, int)) and video_offset >= 0) or \
               ("video_duration" in kwargs and video_offset is None), \
            "Please provide valid value for video_offset parameter"
        self.video_path = video_path
        self.subtitles_path = subtitles_path

        # Either offset needs to be provided to determine delay between recording and official start of video,
        # in order to macht subtitles. It can only be done by providing the official length of the video, and offset
        # will be determiend automtically.
        if video_offset is not None and video_offset >= 0:
            self.video_offset = video_offset
            self.video_duration = None
        else:
            self.video_duration = kwargs.pop("video_duration")
            self.video_offset = None
            assert self.video_duration >= 1, "please provide valid value for either video_offset or video_duration"

        self.extract_annotations = kwargs.pop("extract_annotations", True)

        if self.extract_annotations:
            self.__init_annotation_extract()
        # Number of frames that will be produced
        self.n_frames = kwargs.pop("n_frames", 2250)
        # Number of overlaying frames
        self.overlay_frames = kwargs.pop("overlay_frames", 1)
        assert (self.overlay_frames - 1) % 2 == 0, "Currently only overlay_frames with an odd number are allowed, " \
                                                   "to be able to reuse manual labelling of active_speaker-single_and_voting frame task."
        # Number of frames to skip between each overlay frame
        self.overlay_frames_skip = kwargs.pop("overlay_frames_skip", 1)
        # Number ofr frames to skip until next frame will be used
        self.frame_step = kwargs.pop("frame_step", 15)
        # self.number_of_times_to_upsampling = kwargs.pop("number_of_times_to_upsampling", 0)
        # Fraction to be devided in test, val and train
        self.fractions = kwargs.pop("fractions", [0.75, 0.1, 0.15])
        # This parameter can be used to specify a fixed start point for the video, instead of sampling it randomly.
        self.fixed_start_point = kwargs.pop("fixed_start_point", None)
        # Provide already used start points, to avoid producing different output frames. But keep in mind, it may start
        # still very close off. It may be necessary to adjust the code.
        self.used_start_points = kwargs.pop("used_start_points", [])
        # Define task name, that will affect output directory
        self.task_name = kwargs.pop("task_name", "subtitle_position_boxes")
        # Determine output path according to current directory and task name
        self.output_path = str(pathlib.Path().resolve().joinpath("dataset_processed", self.task_name))
        # Defines if only positioned (according to subtitle file) and middle positioned framed will be returned or
        # another directory will be provided, that contains uninpainted frame and positioned frame as well. Can only be
        # True if extract_middle_and_default is True, too.
        self.extract_middle_and_default = kwargs.pop("extract_middle_and_default", True)
        # used to determine, if starting point of middle positioned subtitles should be inprinted in default image or
        # not. If not set, only positioned frame will be returned.
        self.default_middle = kwargs.pop("default_middle", True)
        # This parameter is used to output the box for the subtitle position in the middle of the subtitle area
        # indicated by the subtitle frame.
        self.dot_middle_of_subtitle_box = kwargs.pop("dot_middle_of_subtitle_box", True)
        self.count_faces = {}
        self.frames_per_step = kwargs.pop("frames_per_step", 1)
        assert isinstance(self.extract_middle_and_default, bool) and isinstance(self.default_middle, bool) and \
               ((self.extract_middle_and_default is True and self.default_middle is True) or
                self.extract_middle_and_default is False)
        assert isinstance(self.output_path, str) and self.output_path != ""
        assert isinstance(self.n_frames, int) and self.n_frames > 0
        assert isinstance(self.frame_step, int) and self.frame_step >= 0
        assert (self.overlay_frames <= 1 and self.frames_per_step >= 1) or (
                self.overlay_frames >= 1 and self.frames_per_step <= 1), "Only overlay ofr frames_per_step can be used. Not both at the same time."

        self.__initialize_output_directories()
        self.initialized = True

    def __initialize_output_directories(self):
        for value in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.output_path, "A", value), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "B", value), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "_B", value), exist_ok=True)
            if self.extract_middle_and_default:
                os.makedirs(os.path.join(self.output_path, "_A", value), exist_ok=True)
                os.makedirs(os.path.join(self.output_path, "__A", value), exist_ok=True)

    @staticmethod
    def __initialize_video_capture(video_path: str, n_frames: int, frame_step: int, video_offset: int = None,
                                   video_duration: int = None, used_start_points: list = [],
                                   fixed_start_point: int = 100) -> cv2.VideoCapture:
        assert video_path != "" and isinstance(video_path, str) and isinstance(n_frames, int) and n_frames > 0 and \
               isinstance(frame_step, int) and frame_step > 0 and \
               (video_offset is not None or video_duration is not None)

        cap = cv2.VideoCapture(video_path)

        # if cap.isOpened() is False:
        #    cap = cv2.VideoCapture(video_path)

        assert cap.isOpened() is True, "Error reading video file"

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # output_size = (frame_height, frame_width)
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if video_offset is None:
            frame_count = int(video_length)
            duration = frame_count / fps
            offset = duration - video_duration
        else:
            offset = video_offset

        if fixed_start_point is None and isinstance(used_start_points, list) and used_start_points is not None:
            need_length = 1 + (n_frames - 1) * frame_step

            if need_length > video_length:
                start = 0
            else:
                if n_frames > 1000 and frame_step > 13:
                    # necessary since not all frames contain a subtitle and while export_only_first_subtitle_box = False
                    # may export multiple frames at once, we need to take care cor export_only_first_subtitle_box = True
                    max_start = video_length - int(need_length * 1.5)
                else:
                    max_start = video_length - need_length
                start = random.randint(0, max_start + 1)
                try_100_times = 0

                while start in used_start_points and try_100_times <= 100:
                    start = random.randint(0, max_start + 1)
                    try_100_times += 1
        else:
            start = fixed_start_point

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        return cap, offset, frame_width, frame_height

    @staticmethod
    def __filter_subtitles_array(position_video_file_milliseconds: float, subtitles_array: np.ndarray,
                                 offset: float) -> np.ndarray:
        time_frame = round(position_video_file_milliseconds / 1000, 2) + offset
        return subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                               (subtitles_array[:, 0].astype(float) <= time_frame)]

    def create_input_data(self):
        import cv2, pathlib, re
        cap, offset, frame_width, frame_height = self.__initialize_video_capture(video_path=self.video_path,
                                                                                 n_frames=self.n_frames,
                                                                                 frame_step=self.frame_step,
                                                                                 video_offset=self.video_offset,
                                                                                 video_duration=self.video_duration,
                                                                                 used_start_points=
                                                                                 self.used_start_points,
                                                                                 fixed_start_point=
                                                                                 self.fixed_start_point)

        subtitles_array = read_in_xml_subtitles(self.subtitles_path)

        GLIP_coco_like_helper = GLIP_coco_like(video_path=self.video_path, subtitles_array=subtitles_array,
                                               used_start_points=[])
        # Create a tqdm progress bar
        progress_bar = tqdm(total=self.n_frames, desc="Processing")

        i = 0
        # current_state_previous = None
        frames = []
        frames_face_recognition = []
        frame_original_msec = None
        subtitles = []
        steps_counter_whole = 0
        temp_frames_per_step = self.frames_per_step
        while i < self.n_frames:
            range_to_skip = self.frame_step - int(((self.overlay_frames - 1) / 2) * self.overlay_frames_skip) - int(
                (self.frames_per_step - 1) / 2)
            if temp_frames_per_step > 0 and steps_counter_whole > 0 and self.frames_per_step > 1:
                range_to_skip = 1
            elif steps_counter_whole > 0 and (self.overlay_frames > 1 or self.frames_per_step > 1):
                range_to_skip -= max(int((self.frames_per_step - 1) / 2), int((self.overlay_frames - 1) / 2))
            # needs to be activated to start sampling directly from start point --> deactivated, since original set was sampled with start + range_to_skip
            # if i == 0:
            #    ret,frame = cap.read()
            # else:
            steps_counter = 0
            # if temp_frames_per_step <= 0 or steps_counter_whole == 0:
            for _ in range(range_to_skip):
                if i < self.n_frames:
                    ret, frame = cap.read()
                else:
                    ret = False
                steps_counter += 1

            if temp_frames_per_step == 0:
                temp_frames_per_step = self.frames_per_step

            if self.overlay_frames > 1 and ret:
                # frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # print(f"Current frame number: {frame_number}")

                temp_frame = frame.copy()
                for step_frames in range((self.overlay_frames - 1)):
                    for step in range(self.overlay_frames_skip):
                        # if steps_counter_whole >= 35 and steps_counter_whole <=38:
                        #    img = Image.fromarray(frame)
                        #    draw = ImageDraw.Draw(img)
                        #    for face_location in list_return[pz]:
                        #        top, right, bottom, left = face_location
                        #        draw.rectangle([left, top, right, bottom], outline="red", width=3)

                        #    img.show()
                        ret, frame = cap.read()
                        steps_counter += 1
                    if step_frames == 0:
                        frame_original = frame.copy()
                        frame_original_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        # frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        # print(f"Current frame number: {frame_number}")
                        # print(i)

                    # frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    # print(f"Current frame number: {frame_number}")

                    temp_frame = cv2.addWeighted(temp_frame, 1, frame.copy(), 0.4, 0)
                    # temp_frame = frame_original.copy()
                frame = temp_frame.copy()
                temp_frame = None
            # else:
            #    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            #    print(f"Current frame number: {frame_number}")
            #    print(i)
            steps_counter_whole += steps_counter
            if ret:
                #         if crop_video:
                #             print("crop_video")
                #             frame = crop(frame)
                #             if round_base(frame.shape[0]) < frame_height or round_base(frame.shape[1]) < frame_width:
                #                 frame_height = round_base(frame.shape[0])
                #                 frame_width = round_base(frame.shape[1])
                if self.overlay_frames > 1:
                    filtered_subtitles_array = self.__filter_subtitles_array(
                        position_video_file_milliseconds=frame_original_msec,
                        subtitles_array=subtitles_array,
                        offset=offset)
                    frame_original_msec = None
                else:
                    filtered_subtitles_array = self.__filter_subtitles_array(
                        position_video_file_milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC),
                        subtitles_array=subtitles_array,
                        offset=offset)
                if len(filtered_subtitles_array) > 0:
                    subtitles.append(filtered_subtitles_array)
                    frames.append(frame.copy())

                    if self.overlay_frames > 1 or (
                            self.frames_per_step > 1 and temp_frames_per_step == statistics.median(
                        list(range(1, self.frames_per_step + 1)))):
                        if self.overlay_frames > 1:
                            frames_face_recognition.append(frame_original.copy())
                        else:
                            frames_face_recognition.append(frame.copy())

                if self.overlay_frames > 1 or self.frames_per_step > 1:
                    len_frames = len(frames_face_recognition)
                else:
                    len_frames = len(frames)

                if len_frames >= min(19, self.n_frames):
                    # list_return = GLIP_coco_like_helper.find_faces([zp[:, :, ::-1] for zp in frames.copy()], subtitles,
                    if self.overlay_frames > 1 or self.frames_per_step > 1:
                        list_return = GLIP_coco_like_helper.find_faces(frames_face_recognition.copy(), subtitles,
                                                                       iteration=i - len(frames) + 1,
                                                                       only_return_detection_result_bool=True)
                    else:
                        list_return = GLIP_coco_like_helper.find_faces(frames.copy(), subtitles,
                                                                       iteration=i - len(frames) + 1,
                                                                       only_return_detection_result_bool=True)

                    if datetime.now().date() > datetime(2024, 1, 12).date():
                        prange = range(0, len_frames)
                    else:
                        prange = range(0, len_frames - 1)
                    for pz in prange:
                        if len(list_return[pz]) > 0:
                            # img = Image.fromarray(frames_face_recognition[pz])
                            # img.show()
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            # img = Image.fromarray(frames[pz])
                            # img.show()
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            if self.frames_per_step > 1:
                                frames_locater = pz * self.frames_per_step
                                frames_list, subtitles_list = frames[
                                                              frames_locater: frames_locater + self.frames_per_step].copy(), subtitles[
                                                                                                                             frames_locater: frames_locater + self.frames_per_step].copy()
                            else:
                                frames_list, subtitles_list = [frames[pz].copy()], [subtitles[pz]]

                            for pPointer in range(0, len(frames_list)):
                                positioned_frame, default_frame, frame_start, start_point, end_point, frame_subtitle = self.__generate_frame_with_position_dot(
                                    frame=frames_list[pPointer].copy(), subtitle_array=subtitles_list[pPointer],
                                    frame_height=frame_height, frame_width=frame_width,
                                    middle=self.default_middle,
                                    dot_middle_of_subtitle_box=self.dot_middle_of_subtitle_box)
                                # if i == 843 or i == "843" or int(i) >= 1330:
                                #    print("stop_id")
                                #    img = Image.fromarray(default_frame)
                                #    draw = ImageDraw.Draw(img)
                                #    for face_location in list_return[pz]:
                                #        top, right, bottom, left = face_location
                                #        draw.rectangle([left, top, right, bottom], outline="red", width=3)

                                #    img.show()
                                # img = Image.fromarray(default_frame)
                                # draw = ImageDraw.Draw(img)
                                # for face_location in list_return[pz]:
                                #     top, right, bottom, left = face_location
                                #     draw.rectangle([left, top, right, bottom], outline="red", width=3)

                                # img.show()
                                assert self.initialized is True

                                if i < self.fractions[0] * self.n_frames:
                                    current_state = "train"
                                elif i >= self.fractions[0] * self.n_frames and i < (
                                        self.fractions[0] + self.fractions[1]) \
                                        * self.n_frames:
                                    current_state = "val"
                                else:
                                    current_state = "test"

                                if not current_state in self.count_faces:
                                    self.count_faces[current_state] = {}
                                self.count_faces[current_state][i] = len(list_return[pz])

                                if self.extract_middle_and_default:
                                    self.write_frames([os.path.join(self.output_path, "A", current_state),
                                                       os.path.join(self.output_path, "B", current_state),
                                                       os.path.join(self.output_path, "_A", current_state),
                                                       os.path.join(self.output_path, "__A", current_state),
                                                       os.path.join(self.output_path, "_B", current_state)],
                                                      str(i) + "_" + str(pPointer) + ".jpg",
                                                      [default_frame, positioned_frame, frames_list[pPointer].copy(),
                                                       frame_start, frame_subtitle])
                                else:
                                    self.write_frames([os.path.join(self.output_path, "A", current_state),
                                                       os.path.join(self.output_path, "B", current_state)],
                                                      str(i) + "_" + str(pPointer) + ".jpg",
                                                      [default_frame, positioned_frame])

                                if self.extract_annotations:
                                    result = self.determine_encode_mask((frame_height, frame_width),
                                                                        start_point, end_point)

                                    self.create_and_append_annotations_segmentation(current_state, frame_height,
                                                                                    frame_width, i,
                                                                                    result, end_point, start_point,
                                                                                    subtitles_list[pPointer][0][0])

                                    self.save_pixelmap(segmentation_counts=result, pkey=current_state, i=i)
                                    # if current_state_previous != current_state and current_state_previous is not None:
                                    #    if self.annotations:
                                    #        self.write_annotatons(current_state)
                                    #        self.annotations = []
                                    #        self.image_annotation_info = []
                                    # current_state_previous = current_state
                            progress_bar.update(1)
                            i += 1
                        # else:
                        #    img = Image.fromarray(default_frame)
                        #    img.show()
                    frames = []
                    subtitles = []
                    frames_face_recognition = []
                temp_frames_per_step -= 1
            else:
                print("End of file, quitted at iteration: " + str(i))
                break
        # Close the tqdm iterator
        if self.annotations != {"train": [], "val": [], "test": []} and isinstance(self.annotations, dict):
            self.write_annotatons()
        progress_bar.close()

    def create_and_append_annotations_segmentation(self, current_state, frame_height, frame_width, i, result, end_point,
                                                   start_point, begin_ts):
        self.image_annotation_info[current_state].append({
            "license": 1,
            "file_name": str(i) + ".jpg",
            "coco_url": "n/a",
            "height": int(frame_height),
            "width": int(frame_width),
            "date_captured": str(datetime.now()),
            "timestamp_begin": str(begin_ts),
            "flickr_url": "n/a",
            "id": i
        })

        dict_segmentation = frPyObjects(result, result["size"][0], result["size"][1])
        self.annotations[current_state].append({
            "segmentation": {
                "counts": dict_segmentation["counts"].decode('utf-8'),
                "size": dict_segmentation["size"]
            }, "area": float(area(dict_segmentation)),
            "iscrowd": 1, "image_id": i, "bbox": [start_point[0], start_point[1],
                                                  end_point[0] - start_point[0],
                                                  end_point[1] - start_point[1]],
            "category_id": [entry['id'] for entry in self.annotation_categories if
                            entry['name'] == "subtitle-position"][0],
            "id": len(self.annotations[current_state])
        })

    def save_pixelmap(self, segmentation_counts: str, pkey: str, i: int):
        colors = {0: 255, 1: 1}  # RGB values
        mask_array = decode(
            frPyObjects(segmentation_counts, segmentation_counts["size"][0], segmentation_counts["size"][1]))

        image = np.zeros((segmentation_counts["size"][0], segmentation_counts["size"][1], 1), dtype=np.uint8)
        # Set pixel values using NumPy indexing
        image[np.where(mask_array == 0)] = colors[0]  # Set dark gray for mask value 0
        image[np.where(mask_array == 1)] = self.annotation_categories[0]['id']  # colors[1]

        # image = image.transpose(image, (1, 0, 2))
        image = np.squeeze(image, axis=2)
        image = Image.fromarray(image, mode="L")

        # image.show()
        os.makedirs(os.path.join(self.output_path, "pixelmaps", pkey), exist_ok=True)
        file_path = os.path.join(self.output_path, "pixelmaps", pkey, str(i) + ".png")
        # image = image.convert("RGB")
        image.save(file_path)

    def write_annotatons(self):
        for pkey, value in self.annotations.items():
            # Specify the file path where you want to save the JSON data
            file_path = os.path.join(self.output_path, "jsons", pkey, "results.json")

            final_dict = {
                "info": self.annotation_info,
                "images": self.image_annotation_info[pkey],
                "licenses": self.annotation_licences,
                "categories": self.annotation_categories,
                "annotations": value
            }

            os.makedirs(os.path.join(self.output_path, "jsons", pkey), exist_ok=True)
            # Write the dictionary to the JSON file
            with open(file_path, 'w') as json_file:
                json.dump(final_dict, json_file)

            if self.count_faces:
                file_path = os.path.join(self.output_path, "jsons", pkey, "count_faces.json")
                with open(file_path, 'w') as json_file:
                    json.dump(self.count_faces[pkey], json_file)

    @staticmethod
    def determine_encode_mask(image_size, start_point, end_point):
        # Create a binary mask based on the provided points
        mask = np.zeros(image_size, dtype=np.uint8)
        # mask = np.zeros_like(image, dtype=np.uint8)
        # cv2.rectangle(mask, start_point, end_point, color=1, thickness=cv2.FILLED)
        x_start, y_start = start_point
        x_end, y_end = end_point
        mask[y_start:y_end, x_start:x_end] = 100

        # from PIL import Image
        # pil_image = Image.fromarray(mask)
        #
        # pil_image.show()
        # exit(-1)

        # mask = mask.transpose()
        # Calculate RLE-encoded counts using vectorized operations
        mask_flattened = mask.reshape(-1, order="F")
        change_positions = np.where(np.diff(mask_flattened) != 0)[0]
        final_array = np.diff(change_positions)

        final_array = np.concatenate(([change_positions[0]], final_array, [len(mask_flattened) - change_positions[-1]]))
        # Create RLE-encoded dictionary
        import matplotlib.pyplot as plt
        from pycocotools import mask as mask_utils

        # dict_segmentation = frPyObjects({'counts': final_array.tolist(), 'size': mask.shape[:2]}, mask.shape[:2][0], mask.shape[:2][1])
        # plt.figure(figsize=(8, 8))
        # plt.imshow(mask_utils.decode(dict_segmentation))
        # plt.axis('off')
        # plt.show()
        rle_encoded = {'counts': final_array.tolist(), 'size': mask.shape[:2]}

        return rle_encoded

    @staticmethod
    def write_frames(pathes: list, name: str, frames: list):
        assert len(pathes) == len(frames)

        for idx, path in enumerate(pathes):
            if not os.path.exists(os.path.join(path, name)):
                cv2.imwrite(path + "/" + name, frames[idx])

    @staticmethod
    def display_image(frame: cv2.GFrame):
        cv2.imshow('Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def __generate_frame_with_position_dot(frame, subtitle_array: np.ndarray, frame_width: int, frame_height: int,
                                           middle: bool = True, size: int = 30,
                                           dot_middle_of_subtitle_box: bool = True):
        # only use first subtitle position as starting point. Probably extend
        subtitle_array = subtitle_array[0]
        if dot_middle_of_subtitle_box:
            x_start_orig = float(subtitle_array[2].replace("%", "")) / 100
            x_end = (float(subtitle_array[4].replace("%", "")) / 100) + x_start_orig
            y_start_orig = float(subtitle_array[3].replace("%", "")) / 100
            y_end = (float(subtitle_array[5].replace("%", "")) / 100) + y_start_orig

            x_start = ((x_start_orig + x_end) / 2) * frame_width
            y_start = ((y_start_orig + y_end) / 2) * frame_height

            x_start_orig = x_start_orig * frame_width
            y_start_orig = y_start_orig * frame_height
        else:
            x_start = (float(subtitle_array[2].replace("%", "")) / 100) * frame_width
            y_start = (float(subtitle_array[3].replace("%", "")) / 100) * frame_height

        frame_positioned = frame.copy()
        frame_positioned = cv2.rectangle(frame_positioned, (round(x_start - (size / 2)), round(y_start - (size / 2))),
                                         (round(x_start + size / 2), round(y_start + size / 2)), (255, 0, 0), -1)

        frame_subtitle = frame.copy()
        frame_subtitle = cv2.putText(frame_subtitle, subtitle_array[6], (int(x_start_orig), int(y_start_orig)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        if middle:
            frame_start = frame.copy()
            frame_start = cv2.rectangle(frame_start,
                                        (round((frame_width / 2) - (size / 2) - (((float(
                                            subtitle_array[4].replace("%", "")) / 100) / 2) * frame_width)),
                                         round(frame_height * 0.9 - (size / 2))),
                                        (round((frame_width / 2) + (size / 2) - (((float(
                                            subtitle_array[4].replace("%", "")) / 100) / 2) * frame_width)),
                                         round(frame_height * 0.9 + (size / 2))),
                                        (255, 0, 0), -1)
            frame = cv2.rectangle(frame,
                                  (round((frame_width / 2) - (size / 2)), round(frame_height * 0.9 - (size / 2))),
                                  (round((frame_width / 2) + (size / 2)), round(frame_height * 0.9 + (size / 2))),
                                  (255, 0, 0), -1)
            return frame_positioned, frame, frame_start, (round(x_start - (size / 2)), round(y_start - (size / 2))), (
                round(x_start + size / 2), round(y_start + size / 2)), frame_subtitle
        else:
            return frame_positioned, frame, (round(x_start - (size / 2)), round(y_start - (size / 2))), (
                round(x_start + size / 2), round(y_start + size / 2)), frame_subtitle


class GLIP_coco_like:
    categories = [{"id": 0, "license": 1, "name": "speaker", "supercategory": "persons"},
                  {"id": 1, "license": 1, "name": "listener", "supercategory": "persons"},
                  {"id": 2, "license": 1, "name": "no_speaker", "supercategory": "persons"}]

    categories_subtitles = [{"id": 0, "license": 1, "name": "Aligned subtitle positioned close to active speaker",
                             "supercategory": "subtitles"},
                            {"id": 1, "license": 1, "name": "Unaligned subtitle positioned in the bottom middle",
                             "supercategory": "subtitles"}
                            ]

    licenses = [{"id": 1, "url": "", "name": "SWAT dataset"}]

    info = {"year": "2023", "version": "1", "description": "Exported from SWAT TV series and subtitles information",
            "contributor": "", "url": "", "date_created": str(datetime.now())}

    def __init__(self, video_path: str, subtitles_array: np.ndarray, used_start_points: list,
                 video_offset: float = None,
                 video_duration: float = None, crop_video: bool = False,
                 n_frames: int = 1000, frame_step: int = 15, face_detection_batch_size: int = 20,
                 number_of_times_to_upsampling: int = 0, test: bool = False, val: bool = False,
                 weight_subtitle_placement_x: list = [2, 1], return_captions: bool = False,
                 return_subtitle_alignment_captions: bool = False,
                 remove_images_without_annotations: bool = False, only_frames: bool = False,
                 overlay_frames: int = 0):

        self.video_path = video_path
        self.subtitles_array = subtitles_array
        self.used_start_points = used_start_points
        self.video_offset = video_offset
        self.video_duration = video_duration
        self.crop_video = crop_video
        self.n_frames = n_frames
        self.frame_step = frame_step
        self.face_detection_batch_size = face_detection_batch_size
        self.number_of_times_upsampling = number_of_times_to_upsampling
        self.test = test
        self.val = val
        self.weight_subtitle_placement_x = weight_subtitle_placement_x
        self.return_captions = return_captions
        self.return_subtitle_alignment_captions = return_subtitle_alignment_captions
        self.overlay_frames = overlay_frames
        self.init_annot = False
        self.images = []
        self.annotations = []
        self.annotations_counter = 1
        self.remove_images_without_annotations = remove_images_without_annotations
        self.only_frames = only_frames

        if return_captions:
            self.captions_counter = 1
            self.captions = []
            self.init_capt = False

    def __call__(self, *args, **kwargs):
        self.create_input_data_GLIP(video_path=self.video_path, subtitles_array=self.subtitles_array,
                                    used_start_points=self.used_start_points, video_offset=self.video_offset,
                                    video_duration=self.video_duration, crop_video=self.crop_video,
                                    n_frames=self.n_frames, frame_step=self.frame_step,
                                    face_detection_batch_size=self.face_detection_batch_size,
                                    number_of_times_to_upsampling=self.number_of_times_upsampling,
                                    test=self.test, weight_subtitle_placement_x=self.weight_subtitle_placement_x,
                                    val=self.val, return_captions=self.return_captions,
                                    return_subtitle_alignment_captions=self.return_subtitle_alignment_captions,
                                    remove_images_without_annotations=self.remove_images_without_annotations,
                                    only_frames=self.only_frames, overlay_frames=self.overlay_frames)
        return self

    def find_faces(self, frames: list, subtitles: list, iteration: int = 0, number_of_times_to_upsample: int = 0,
                   frame_width: int = 1920, frame_height: int = 1080,
                   weight_subtitle_placement_x: list = [2, 1], only_return_detection_result_bool=False) -> list:
        assert isinstance(frames, list) and len(frames) > 0, "Please provide valid list for input parameter frames"

        import face_recognition

        others_dict = {}
        closests_dict = {}
        complete_dict = {}
        subtitles_dict = {}
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=
        number_of_times_to_upsample, batch_size=19)

        if only_return_detection_result_bool:
            return batch_of_face_locations
        # Now let's list all the faces we found in all frames
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_number_in_batch + iteration
            begin, end, origin_x, origin_y, extent_x, extent_y, text, region = subtitles

            # print(subtitles)

            # print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))

            idx_min = -1000
            dist_min = 10000000
            if len(face_locations) == 0:
                if frame_number in complete_dict:
                    (complete_dict[frame_number]).append([])
                else:
                    complete_dict[frame_number] = [[]]
            for pidx, face_location in enumerate(face_locations):

                top, left, bottom, right = face_location

                additional_a = int(frame_width * (float(origin_x.strip("%")) / 100)), \
                    int(frame_height * (float(origin_y.strip("%")) / 100)), \
                    int(frame_width * ((float(origin_x.strip("%")) + float(extent_x.strip("%"))) / 100)), \
                    int(frame_height * ((float(origin_y.strip("%")) + float(extent_y.strip("%"))) / 100))

                a = np.array([(additional_a[0] * weight_subtitle_placement_x[0] + additional_a[2] *
                               weight_subtitle_placement_x[1]) / sum(weight_subtitle_placement_x),
                              (additional_a[1] + additional_a[3]) / 2])

                # a = np.array([int(frame_width * ((float(origin_x.strip("%")) / 100) + (
                #        (float(origin_x.strip("%")) + float(extent_x.strip("%"))) / 100)) / 2),
                #              int(frame_height * ((float(origin_y.strip("%")) / 100) + (
                #                      (float(origin_y.strip("%")) + float(extent_y.strip("%"))) / 100)) / 2)])

                subtitles_dict[frame_number] = [a, additional_a]

                b = np.array([(left + right) / 2, (top + bottom) / 2])

                if np.linalg.norm(a - b) < dist_min and pidx != idx_min:
                    dist_min = np.linalg.norm(a - b)
                    idx_min = pidx

            face_locations = np.array(face_locations)
            if idx_min >= 0 or len(face_locations) > 0:
                mask = np.ones(len(face_locations), dtype=bool)
                mask[idx_min] = False
                for pfl in face_locations[mask]:
                    # print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                    if frame_number in others_dict:
                        (others_dict[frame_number]).append(pfl)
                    else:
                        others_dict[frame_number] = [pfl]

                    if frame_number in complete_dict:
                        (complete_dict[frame_number]).append(pfl.tolist())
                    else:
                        complete_dict[frame_number] = [pfl.tolist()]

                for pfl in face_locations[~mask]:
                    # print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                    if frame_number in closests_dict:
                        (closests_dict[frame_number]).append(pfl)
                    else:
                        closests_dict[frame_number] = [pfl]

                    if frame_number in complete_dict:
                        (complete_dict[frame_number]).append(pfl.tolist())
                    else:
                        complete_dict[frame_number] = [pfl.tolist()]
        # return final_list
        return complete_dict, closests_dict, others_dict, subtitles_dict

    def create_input_data_GLIP(self, video_path: str, subtitles_array: np.ndarray, used_start_points: list,
                               weight_subtitle_placement_x: list, video_offset: float = None,
                               video_duration: float = None, crop_video: bool = False,
                               n_frames: int = 1000, frame_step: int = 15, face_detection_batch_size: int = 20,
                               number_of_times_to_upsampling: int = 0, test: bool = False, val: bool = False,
                               return_captions: bool = False, return_subtitle_alignment_captions: bool = False,
                               remove_images_without_annotations: bool = False, only_frames: bool = False,
                               overlay_frames: int = 0):

        import cv2, random, pathlib, re

        assert video_path is not None and video_path != "", "Please provide proper video path"
        assert video_offset is not None or video_duration is not None, 'Please provide either ' \
                                                                       '--vo or --vd command ' \
                                                                       'line parameter'
        assert not all([test, val]), "Please only provide val or test parameter as True"
        assert only_frames and overlay_frames >= 0 or only_frames is False and overlay_frames <= 0, "Parameter " \
                                                                                                    "overlay_pictures " \
                                                                                                    "can only be used " \
                                                                                                    "if only_frames is " \
                                                                                                    "set to true"
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened() is False:
            cap = cv2.VideoCapture(video_path.replace("OneDrive - IUBH Internationale Hochschule",
                                                      "OneDrive - IU International University of Applied Sciences"))
        assert cap.isOpened() is True, "Error reading video file"

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_size = (frame_height, frame_width)
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if video_offset is None:
            frame_count = int(video_length)
            duration = frame_count / fps
            offset = duration - video_duration
        else:
            offset = video_offset

        need_length = 1 + (n_frames - 1) * frame_step

        if need_length > video_length:
            start = 0
        else:
            if n_frames > 1000 and frame_step > 15:
                # necessary since not all frames contain a subtitle and while export_only_first_subtitle_box = False
                # may export multiple frames at once, we need to take care cor export_only_first_subtitle_box = True
                max_start = video_length - int(need_length * 1.3)
            else:
                max_start = video_length - need_length
            start = random.randint(0, max_start + 1)
            try_100_times = 0
            while start in used_start_points and try_100_times <= 100:
                start = random.randint(0, max_start + 1)
                try_100_times += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        name_fixed = str(pathlib.Path().resolve().joinpath("dataset_processed", re.sub(r" +", "_",
                                                                                       video_path.split("/")[-1].split(
                                                                                           ".")[
                                                                                           0].replace(
                                                                                           "-", ""))))
        if test:
            name_fixed = name_fixed.replace("_train", "_test")
        if val:
            name_fixed = name_fixed.replace("_train", "_val")
        i = 0

        # if only_frames is False:
        batch_frames = []
        if overlay_frames > 0:
            overlap_counter = 0
            frame_temp = None
        while i < n_frames:
            # if i >= n_frames:
            #    break
            for _ in range(frame_step):
                if i < n_frames:
                    ret, frame = cap.read()
                else:
                    ret = False
            if ret:
                if crop_video:
                    print("crop_video")
                    frame = crop(frame)
                    if round_base(frame.shape[0]) < frame_height or round_base(frame.shape[1]) < frame_width:
                        frame_height = round_base(frame.shape[0])
                        frame_width = round_base(frame.shape[1])
                        # output_size = (frame_width, frame_height)

                # cv2.putText(frame, text, (int(frame_width * (float(origin_x.strip("%")) / 100)),
                #                          int(frame_height * (float(origin_y.strip("%")) / 100))),
                #            cv2.FONT_HERSHEY_SIMPLEX, 2.25, (0, 255, 0))

                time_frame = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2) + offset

                if len(subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                                       (subtitles_array[:, 0].astype(float) <= time_frame)]) > 0:
                    if overlay_frames > 0:
                        if overlap_counter <= overlay_frames:
                            if overlap_counter == 0 or frame_temp is None:
                                frame_temp = frame
                            else:
                                frame_temp = cv2.addWeighted(frame_temp, 1, frame, 0.4, 0)
                                # frame_temp = np.concatenate((frame_temp, frame), axis=0)
                            overlap_counter += 1
                        else:
                            frame_temp = None
                            overlap_counter = 0
                    else:
                        overlap_counter = -100

                    if only_frames is False and return_subtitle_alignment_captions is False:
                        batch_frames.append(frame[:, :, ::-1])

                    if face_detection_batch_size > 0:
                        modulo_result = len(batch_frames) % face_detection_batch_size == 0
                    else:
                        modulo_result = False
                    if (modulo_result and i > 0 and batch_frames != []) or (only_frames and (
                            overlap_counter == overlay_frames) or overlay_frames == 0) or return_subtitle_alignment_captions:

                        pass_subtitle_list = subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                                                             (subtitles_array[:, 0].astype(float) <= time_frame)][0]

                        if only_frames is False and return_subtitle_alignment_captions is False:
                            # pass_subtitle_list = subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                            #                                     (subtitles_array[:, 0].astype(float) <= time_frame)][0]
                            complete_dict, closest_dict, others_dict, subtitles_dict = self.find_faces(
                                frames=batch_frames, iteration=(i - len(batch_frames) + 1),
                                number_of_times_to_upsample=number_of_times_to_upsampling, subtitles=pass_subtitle_list,
                                frame_width=frame_width, frame_height=frame_height,
                                weight_subtitle_placement_x=weight_subtitle_placement_x)

                        if only_frames is False or return_subtitle_alignment_captions:
                            name_b = name_fixed + "\\B\\"

                        name_a = name_fixed + "\\A\\"

                        if only_frames is False and return_subtitle_alignment_captions is False:
                            file_names = self.write_frames(frames=batch_frames, annotations=[closest_dict, others_dict],
                                                           pathes=[name_a, name_b], i=i, subtitles_dict=subtitles_dict,
                                                           only_frames=only_frames)
                        elif return_subtitle_alignment_captions and overlay_frames < 1:
                            subtitle_annotations = self.get_annotations_subtitle_boxes(pass_subtitle_list,
                                                                                       [frame_width, frame_height], i)

                            file_names = self.write_frames(frames=[frame], annotations=subtitle_annotations,
                                                           pathes=[name_a, name_b], i=i, subtitles_dict={},
                                                           only_frames=only_frames, subtitles_annotations=True)
                        elif overlay_frames < 1:
                            file_names = self.write_frames(frames=[frame],
                                                           pathes=[name_a], i=i,
                                                           only_frames=only_frames)
                        else:
                            file_names = self.write_frames(frames=[frame_temp],
                                                           pathes=[name_a], i=i,
                                                           only_frames=only_frames)
                        if only_frames is False:
                            self.images.extend(file_names)
                            # del batch_frames
                            if return_subtitle_alignment_captions:
                                self.annotations.extend(subtitle_annotations)
                            else:
                                if return_captions:
                                    annotations, captions = self.get_annotations_face_recognition(
                                        complete_dict=complete_dict,
                                        closest_dict=closest_dict,
                                        others_dict=others_dict,
                                        return_captions=True)
                                else:
                                    annotations = self.get_annotations_face_recognition(complete_dict=complete_dict,
                                                                                        closest_dict=closest_dict,
                                                                                        others_dict=others_dict,
                                                                                        return_captions=False)

                                if annotations:
                                    self.annotations.extend(annotations)

                                if 'captions' in vars():
                                    if captions:
                                        self.captions.extend(captions)

                                # write_face_annotations(complete_dict,
                                #                       name_fixed.replace("_unpositioned", "").replace("_positioned",
                                #                                                                       "") + "frames.json", i)
                                batch_frames = []

                    if overlay_frames is False or overlay_frames == 0 or overlap_counter == 0:
                        i = i + 1

            if i >= n_frames:
                break

        if len(batch_frames) > 0 and only_frames is False:
            pass_subtitle_list = subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                                                 (subtitles_array[:, 0].astype(float) <= time_frame)][0]

            complete_dict, closest_dict, others_dict, subtitles_dict = self.find_faces(frames=batch_frames,
                                                                                       iteration=(i - len(
                                                                                           batch_frames) + 1),
                                                                                       number_of_times_to_upsample=number_of_times_to_upsampling,
                                                                                       subtitles=pass_subtitle_list,
                                                                                       frame_width=frame_width,
                                                                                       frame_height=frame_height,
                                                                                       weight_subtitle_placement_x=weight_subtitle_placement_x)
            name_a = name_fixed + "\\A\\"
            name_b = name_fixed + "\\B\\"

            file_names = self.write_frames(frames=batch_frames, annotations=[closest_dict, others_dict],
                                           pathes=[name_a, name_b], i=i,
                                           subtitles_dict=subtitles_dict)

            self.images.extend(file_names)
            # del batch_frames
            if return_captions:
                annotations, captions = self.get_annotations_face_recognition(complete_dict=complete_dict,
                                                                              closest_dict=closest_dict,
                                                                              others_dict=others_dict,
                                                                              return_captions=True)
            else:
                annotations = self.get_annotations_face_recognition(complete_dict=complete_dict,
                                                                    closest_dict=closest_dict,
                                                                    others_dict=others_dict,
                                                                    return_captions=False)

            if annotations:
                self.annotations.extend(annotations)

            if captions:
                self.captions.extend(captions)
            # del batch_frames
            # write_face_annotations(complete_list,
            #                       name_fixed.replace("_unpositioned", "").replace("_positioned",
            #                                                                       "") + "frames.json", i)
            del batch_frames

        self.write_face_annotations(path=name_fixed.replace("_unpositioned", "").replace("_positioned",
                                                                                         "") + "frames.json",
                                    write_captions=return_captions,
                                    path_captions=name_fixed.replace("_unpositioned", "").replace("_positioned",
                                                                                                  "") + "frames_captions.json",
                                    remove_images_without_annotations=remove_images_without_annotations)

    def get_annotations_subtitle_boxes(self, subtitles: np.array, resolution: list, frame_no: int):
        assert isinstance(subtitles, np.ndarray)
        assert isinstance(resolution, list)
        assert frame_no > -1

        final_list = []
        caption = ""
        if subtitles.ndim == 1:
            start_x = subtitles[2]
            start_y = subtitles[3]
            extent_x = subtitles[4]
            extent_y = subtitles[5]

            x_start = (float(start_x.replace("%", "")) / 100) * resolution[0]
            y_start = (float(start_y.replace("%", "")) / 100) * resolution[1]
            extent_x = (float(extent_x.replace("%", "")) / 100) * resolution[0]
            extent_y = (float(extent_y.replace("%", "")) / 100) * resolution[1]

            # x_end = (start_x + extent_x) * resolution[0]
            # y_end = (start_y + extent_y) * resolution[0]

            # final_list.append([x_start, y_start, x_end, y_end])
            category_id = next((item for item in self.categories_subtitles if "active speaker" in item["name"]), None)[
                "id"]
            final_list.append({"id": self.annotations_counter, "image_id": frame_no, "category_id": category_id,
                               "bbox": [x_start, y_start, extent_x, extent_y],
                               "area": (extent_x * extent_y), "segmentation": [], "iscrowd": 0})
            self.annotations_counter += 1

            category_id = next((item for item in self.categories_subtitles if "bottom middle" in item["name"]), None)[
                "id"]
            final_list.append({"id": self.annotations_counter, "image_id": frame_no, "category_id": category_id,
                               "bbox": [(resolution[0] - extent_x) / 2, (resolution[1] - extent_y - 50), extent_x,
                                        extent_y],
                               "area": (extent_x * extent_y), "segmentation": [], "iscrowd": 0})
            self.annotations_counter += 1
        else:
            for time_start, time_end, start_x, start_y, extent_x, extent_y, text, category in subtitles:
                x_start = (float(start_x.replace("%", "")) / 100) * resolution[0]
                y_start = (float(start_y.replace("%", "")) / 100) * resolution[1]

                extent_x = (float(extent_x.replace("%", "")) / 100) * resolution[0]
                extent_y = (float(extent_y.replace("%", "")) / 100) * resolution[1]

                # x_end = (start_x + extent_x) * resolution[0]
                # y_end = (start_y + extent_y) * resolution[0]

                # final_list.append([x_start, y_start, x_end, y_end])
                category_id = \
                    next((item for item in self.categories_subtitles if "active speaker" in item["name"]), None)["id"]
                final_list.append({"id": self.annotations_counter, "image_id": frame_no, "category_id": category_id,
                                   "bbox": [x_start, y_start.item(), extent_x, extent_y],
                                   "area": (extent_x * extent_y), "segmentation": [], "iscrowd": 0})
                self.annotations_counter += 1

                category_id = \
                    next((item for item in self.categories_subtitles if "bottom middle" in item["name"]), None)["id"]
                final_list.append({"id": self.annotations_counter, "image_id": frame_no, "category_id": category_id,
                                   "bbox": [(resolution[0] - extent_x) / 2, (resolution[1] - extent_y - 50), extent_x,
                                            extent_y],
                                   "area": (extent_x * extent_y), "segmentation": [], "iscrowd": 0})
                self.annotations_counter += 1

        return final_list

    def get_annotations_face_recognition(self, complete_dict: dict, closest_dict: dict, others_dict: dict,
                                         return_captions: bool = False) -> list:
        # elements = len(batch_frames)
        return_list = []
        if return_captions:
            return_caption_list = []

        for key, element in complete_dict.items():
            iscrowd = 1 if key in others_dict and key in closest_dict else 0

            if return_captions:
                caption = ''

            if key in closest_dict:
                top, right, bottom, left = closest_dict[key][0]
                category_id = next((item for item in self.categories if item["name"] == "speaker"), None)["id"]
                if return_captions:
                    caption = caption + 'Speaker.'

                return_list.append({"id": self.annotations_counter, "image_id": key, "category_id": category_id,
                                    "bbox": [left.item(), top.item(), (right.item() - left.item()),
                                             (bottom.item() - top.item())],
                                    "area": (right.item() - left.item()) * (bottom.item() - top.item()),
                                    "segmentation": [], "iscrowd": iscrowd})

                self.annotations_counter += 1

            if key in others_dict:
                category_id = next((item for item in self.categories if item["name"] == "listener"), None)["id"]
                if isinstance(others_dict[key], list) and len(others_dict[key]) > 1:
                    for bbox in others_dict[key]:
                        top, right, bottom, left = bbox
                        if return_captions:
                            caption = caption + 'Listener.'
                        return_list.append({"id": self.annotations_counter, "image_id": key, "category_id": category_id,
                                            "bbox": [left.item(), top.item(), (right.item() - left.item()),
                                                     (bottom.item() - top.item())],
                                            "area": (right.item() - left.item()) * (bottom.item() - top.item()),
                                            "segmentation": [],
                                            "iscrowd": iscrowd})
                        self.annotations_counter += 1
                else:
                    if return_captions:
                        caption = caption + 'Listener.'
                    top, right, bottom, left = others_dict[key][0]
                    return_list.append({"id": self.annotations_counter, "image_id": key, "category_id": category_id,
                                        "bbox": [left.item(), top.item(), (right.item() - left.item()),
                                                 (bottom.item() - top.item())],
                                        "area": (right.item() - left.item()) * (bottom.item() - top.item()),
                                        "segmentation": [], "iscrowd": iscrowd})
                    self.annotations_counter += 1

            if return_captions and caption != "":
                return_caption_list.append({"image_id": key, "id": self.captions_counter, "caption": caption})
                self.captions_counter += 1

        if return_captions:
            return return_list, return_caption_list
        else:
            return return_list

    def write_face_annotations(self, path, remove_images_without_annotations: bool = True, write_captions: bool = False,
                               path_captions: str = None):

        assert write_captions is False or (write_captions is True and path_captions is not None and
                                           path_captions != "" and isinstance(path_captions, str)), "Please provide " \
                                                                                                    "write_captions and " \
                                                                                                    "path_captions or " \
                                                                                                    "none of it"
        import json
        # if isinstance(self.annotations, np.ndarray):
        #    to_append_list = self.annotations.to_list()
        # with open(name_fixed.replace("_unpositioned", "").replace("_positioned", "") + "frames.json", 'w') as fp:

        final_dict = {"info": self.info, "licenses": self.licenses, "categories": self.categories,
                      "images": self.images, "annotations": self.annotations}

        if write_captions:
            final_captions_dict = {"info": self.info, "licenses": self.licenses,
                                   "images": self.images, "annotations": self.captions}

        if remove_images_without_annotations:
            image_ids = []

            if write_captions:
                enum = self.annotations + self.captions
            else:
                enum = self.annotations

            for dictionary in enum:
                image_ids.append(dictionary["image_id"])

            image_ids = list(set(image_ids))

            # delete_idxs = []
            for idx, dictionary in enumerate(self.images):
                if not dictionary["id"] in image_ids:
                    del self.images[idx]
                    # print(dictionary["id"])
                # delete_idxs

        # print(final_dict)
        if self.init_annot is False and len(self.annotations) > 0:
            with open(path, 'w') as fp:
                json.dump(final_dict, fp)
            self.init_annot = True
        else:
            with open(path, 'a') as fp:
                json.dump(final_dict, fp)

        if write_captions:
            if self.init_capt is False and len(self.captions) > 0:
                with open(path_captions, 'w') as fp:
                    json.dump(final_captions_dict, fp)
                self.init_capt = True
            else:
                with open(path_captions, 'a') as fp:
                    json.dump(final_captions_dict, fp)

    def write_frames(self, frames: list, pathes: list, i: int, subtitles_dict: dict = {}, annotations: list = [],
                     frame_width: int = 1920, frame_height: int = 1080, only_frames: bool = False,
                     subtitles_annotations: bool = False):
        import cv2
        # test comment

        assert isinstance(frames, (list, np.ndarray)), "Please provide valid frame array"
        assert isinstance(annotations, list) and (
                len(annotations) == 2 or annotations == [] or subtitles_dict == {}), "Please provide annotations as list = " \
                                                                                     "[closest, others]"
        assert isinstance(pathes, list) and ((len(pathes) == 2 and (only_frames is False and len(
            pathes) == 1 and subtitles_annotations is False)) or subtitles_annotations is True), "Please provide pathes as list = " \
                                                                                                 "[original_output, framed_output]"

        file_names = []
        if only_frames is False and subtitles_annotations is False:
            red_annot, green_annot = annotations
        # elif subtitles_annotations:
        #    subtitles_anot = annotations[0]

        for idx, frame in enumerate(frames):
            if subtitles_annotations is False:
                frame = frame[:, :, ::-1]
            # face_image = image[top:bottom, left:right]
            cv2.imwrite(pathes[0] + f"{i + idx - len(frames) + 1}.jpg", frame)
            if only_frames is False:
                file_names.append({"id": i + idx - len(frames) + 1, "license": 1,
                                   "file_name": pathes[0] + f"{i + idx - len(frames) + 1}.jpg",
                                   "height": int(frame_height),
                                   "width": int(frame_width), "date_captured": str(datetime.now())})
                if subtitles_annotations is False:
                    if i + idx - len(frames) + 1 in subtitles_dict:
                        if len(subtitles_dict[i + idx - len(frames) + 1]) == 2:
                            x_middle, y_middle = subtitles_dict[i + idx - len(frames) + 1][0].astype(int)
                            x_start, y_start, x_end, y_end = subtitles_dict[i + idx - len(frames) + 1][1]
                            frame = cv2.rectangle(frame, (int(x_middle) - 5, y_middle - 5),
                                                  (x_middle + 5, y_middle + 5),
                                                  (255, 0, 0), -1)
                            frame = cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0),
                                                  1)
                        else:
                            x_middle, y_middle = subtitles_dict[i + idx - len(frames) + 1]
                            frame = cv2.rectangle(frame, (x_middle - 5, y_middle - 5), (x_middle + 5, y_middle + 5),
                                                  (255, 0, 0), -1)

                    if i + idx - len(frames) + 1 in red_annot or i + idx - len(frames) + 1 in green_annot:
                        # print(annotations[i+idx-len(frames)+1])
                        if i + idx - len(frames) + 1 in green_annot:
                            for elem_annot in green_annot[i + idx - len(frames) + 1]:
                                top, right, bottom, left = elem_annot
                                frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

                        if i + idx - len(frames) + 1 in red_annot:
                            for elem_annot in red_annot[i + idx - len(frames) + 1]:
                                top, right, bottom, left = elem_annot
                                frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

                        cv2.imwrite(pathes[1] + f"{i + idx - len(frames) + 1}.jpg", frame)
                    else:
                        cv2.imwrite(pathes[1] + f"{i + idx - len(frames) + 1}.jpg", frame)
                else:
                    x_start, y_start, x_end, y_end = annotations[0][
                        "bbox"]  # annotations[i + idx - len(frames) + 1]["bbox"]
                    frame = cv2.rectangle(frame, (int(x_start), int(y_start)),
                                          (int(x_start + x_end), int(y_start + y_end)), (0, 255, 0),
                                          1)
                    x_start, y_start, x_end, y_end = annotations[1]["bbox"]
                    frame = cv2.rectangle(frame, (int(x_start), int(y_start)),
                                          (int(x_start + x_end), int(y_start + y_end)), (255, 0, 0),
                                          1)
                    cv2.imwrite(pathes[1] + f"{i + idx - len(frames) + 1}.jpg", frame)

        return file_names


# def create_input_data_GLIP(video_path: str, subtitles_array: np.ndarray, used_start_points: list, video_offset: float = None,
#                             video_duration: float = None, crop_video: bool = False,
#                             n_frames: int = 1000, frame_step: int = 15, face_detection_batch_size: int = 20,
#                             number_of_times_to_upsampling: int = 0):
#
#     import cv2, random, pathlib, re
#
#     assert video_path is not None and video_path != "", "Please provide proper video path"
#     assert video_offset is not None or video_duration is not None, 'Please provide either ' \
#                                                                    '--vo or --vd command ' \
#                                                                    'line parameter'
#
#     cap = cv2.VideoCapture(video_path)
#
#     if cap.isOpened() is False:
#         cap = cv2.VideoCapture(video_path.replace("OneDrive - IUBH Internationale Hochschule",
#                                                   "OneDrive - IU International University of Applied Sciences"))
#     assert cap.isOpened() is True, "Error reading video file"
#
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     output_size = (frame_height, frame_width)
#     video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if video_offset is None:
#         frame_count = int(video_length)
#         duration = frame_count / fps
#         offset = duration - video_duration
#     else:
#         offset = video_offset
#
#     need_length = 1 + (n_frames - 1) * frame_step
#
#     if need_length > video_length:
#         start = 0
#     else:
#         if n_frames > 1000 and frame_step > 15:
#             # necessary since not all frames contain a subtitle and while export_only_first_subtitle_box = False
#             # may export multiple frames at once, we need to take care cor export_only_first_subtitle_box = True
#             max_start = video_length - int(need_length * 1.3)
#         else:
#             max_start = video_length - need_length
#         start = random.randint(0, max_start + 1)
#         try_100_times = 0
#         while start in used_start_points and try_100_times <= 100:
#             start = random.randint(0, max_start + 1)
#             try_100_times += 1
#
#     print(start)
#
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start)
#
#     name_fixed = str(pathlib.Path().resolve().joinpath("data", re.sub(r" +", "_",
#                                                                       video_path.split("/")[-1].split(".")[0].replace(
#                                                                           "-", ""))))
#     i = 0
#     batch_frames = []
#
#     while i < n_frames:
#         # if i >= n_frames:
#         #    break
#         for _ in range(frame_step):
#             if i < n_frames:
#                 ret, frame = cap.read()
#             else:
#                 ret = False
#         if ret:
#             if crop_video:
#                 frame = crop(frame)
#                 if round_base(frame.shape[0]) < frame_height or round_base(frame.shape[1]) < frame_width:
#                     frame_height = round_base(frame.shape[0])
#                     frame_width = round_base(frame.shape[1])
#                     output_size = (frame_width, frame_height)
#
#             # cv2.putText(frame, text, (int(frame_width * (float(origin_x.strip("%")) / 100)),
#             #                          int(frame_height * (float(origin_y.strip("%")) / 100))),
#             #            cv2.FONT_HERSHEY_SIMPLEX, 2.25, (0, 255, 0))
#
#             time_frame = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2) + offset
#
#
#             if len(subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
#                                 (subtitles_array[:, 0].astype(float) <= time_frame)]) > 0:
#
#
#                 batch_frames.append(frame[:, :, ::-1])
#                 if len(batch_frames) % face_detection_batch_size == 0 and i > 0:
#
#                     pass_subtitle_list = subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
#                                           (subtitles_array[:, 0].astype(float) <= time_frame)][0]
#                     complete_dict, closest_dict, others_dict, subtitles_dict = find_faces(frames=batch_frames, iteration=(i - len(batch_frames) + 1),
#                                                 number_of_times_to_upsample=number_of_times_to_upsampling,
#                                                 subtitles=pass_subtitle_list)
#                     name_a = name_fixed + r"\A\\"
#                     name_b = name_fixed + r"\B\\"
#
#                     write_frames(batch_frames, [closest_dict, others_dict], [name_a, name_b], i, subtitles_dict)
#                     #del batch_frames
#                     write_face_annotations(complete_dict,
#                                            name_fixed.replace("_unpositioned", "").replace("_positioned",
#                                                                                            "") + "frames.json", i)
#                     batch_frames = []
#                 i = i + 1
#
#         if i >= n_frames:
#             break
#
#     if len(batch_frames) > 0:
#         pass_subtitle_list = subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
#                                              (subtitles_array[:, 0].astype(float) <= time_frame)][0]
#         complete_list, closest_list, others_list = find_faces(frames=batch_frames,
#                                                               iteration=(i - len(batch_frames) + 1),
#                                                               number_of_times_to_upsample=number_of_times_to_upsampling,
#                                                               subtitles=pass_subtitle_list)
#         name_a = name_fixed + r"\A\\"
#         name_b = name_fixed + r"\B\\"
#         write_frames(batch_frames, [closest_list, others_list], [name_a, name_b], i)
#         # del batch_frames
#         write_face_annotations(complete_list,
#                                name_fixed.replace("_unpositioned", "").replace("_positioned",
#                                                                                "") + "frames.json", i)
#         del batch_frames

def create_input_data(video_path: str, subtitles_array: np.ndarray, used_start_points: list, video_offset: float = None,
                      video_duration: float = None, safe_training_data: bool = False, crop_video: bool = False,
                      n_frames: int = 1000, frame_step: int = 15, return_frames: bool = True,
                      position_subtitle_middle: bool = False, output_frames_instructions: bool = False,
                      output_frames_instructions_middle_positioned: bool = False, output_box: bool = True,
                      export_only_first_subtitle_box: bool = False, export_frame_positions: bool = True,
                      number_of_times_to_upsampling: int = 0):
    import cv2
    import random
    # import tensorflow as tf
    # reading the input
    import pathlib
    import re

    assert not (position_subtitle_middle and output_frames_instructions_middle_positioned), "only set " \
                                                                                            "position_subtitle_middle " \
                                                                                            "or output_frame_instructions_middle_positioned " \
                                                                                            "to True"
    assert not (position_subtitle_middle and output_box), "only set position_subtitle_middle or output_box to True"
    assert not (output_frames_instructions_middle_positioned and output_box), "only set " \
                                                                              "output_frames_instructions_middle_positioned to True"
    assert video_path is not None and video_path != "", "Please provide proper video path"
    assert video_offset is not None or video_duration is not None, 'Please provide either ' \
                                                                   '--vo or --vd command ' \
                                                                   'line parameter'
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened() is False:
        cap = cv2.VideoCapture(video_path.replace("OneDrive - IUBH Internationale Hochschule",
                                                  "OneDrive - IU International University of Applied Sciences"))
    assert cap.isOpened() is True, "Error reading video file"

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_size = (frame_height, frame_width)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if video_offset is None:
        frame_count = int(video_length)
        duration = frame_count / fps
        offset = duration - video_duration
    else:
        offset = video_offset

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        if export_only_first_subtitle_box is True and n_frames > 1000 and frame_step > 15:
            # necessary since not all frames contain a subtitle and while export_only_first_subtitle_box = False
            # may export multiple frames at once, we need to take care cor export_only_first_subtitle_box = True
            max_start = video_length - int(need_length * 1.3)
        else:
            max_start = video_length - need_length
        start = random.randint(0, max_start + 1)
        try_100_times = 0
        while start in used_start_points and try_100_times <= 100:
            start = random.randint(0, max_start + 1)
            try_100_times += 1

    # print(start)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    # output = cv2.VideoWriter(
    #    "output.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
    #    25, (frame_width, frame_height))
    # 30, (1080, 1920))

    i = 0
    final_list_frames = []
    final_list_frames_original = []
    if return_frames is False and output_frames_instructions is False:
        final_list_input = []
        final_list_target = []
    elif output_frames_instructions and return_frames is False:
        json_dict_array = []
    else:
        final_list_captions = []
    name_fixed = str(pathlib.Path().resolve().joinpath("dataset_processed", re.sub(r" +", "_",
                                                                                   video_path.split("/")[-1].split(".")[
                                                                                       0].replace(
                                                                                       "-", ""))))
    # if output_frames_instructions_middle_positioned

    if position_subtitle_middle and output_frames_instructions_middle_positioned is False and output_box is False:
        name_fixed = name_fixed + r"\unpositioned_"
    elif position_subtitle_middle is False and output_frames_instructions_middle_positioned is False and output_box is False:
        name_fixed = name_fixed + r"\positioned_"

    while i < n_frames:
        # if i >= n_frames:
        #    break
        for _ in range(frame_step):
            if i < n_frames:
                ret, frame = cap.read()
            else:
                ret = False
        if ret:
            if crop_video:
                frame = crop(frame)
                if round_base(frame.shape[0]) < frame_height or round_base(frame.shape[1]) < frame_width:
                    frame_height = round_base(frame.shape[0])
                    frame_width = round_base(frame.shape[1])
                    output_size = (frame_width, frame_height)

            # cv2.putText(frame, text, (int(frame_width * (float(origin_x.strip("%")) / 100)),
            #                          int(frame_height * (float(origin_y.strip("%")) / 100))),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2.25, (0, 255, 0))

            time_frame = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2) + offset

            count_subtitles = 0
            if output_frames_instructions_middle_positioned or output_box:
                frame_middle = frame.copy()

            for begin, end, origin_x, origin_y, extent_x, extent_y, text, region in \
                    subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
                                    (subtitles_array[:, 0].astype(float) <= time_frame)]:
                # final_list_input.append([(frame.astype(np.float16)/255), len(text), region])
                # final_list_input.append([frame, int(len(text)), int(region)])
                if count_subtitles >= 1 and export_only_first_subtitle_box is True:
                    break
                if return_frames or output_frames_instructions:
                    if export_frame_positions:
                        final_list_frames_original.append(frame[:, :, ::-1])
                        if i % 20 == 0 and i > 0:
                            frame_positions = find_faces(final_list_frames_original,
                                                         (i - len(final_list_frames_original) + 1),
                                                         number_of_times_to_upsample=number_of_times_to_upsampling)
                            write_face_annotations(frame_positions,
                                                   name_fixed.replace("_unpositioned", "").replace("_positioned",
                                                                                                   "") + "frames.json",
                                                   i)
                            final_list_frames_original = []
                    if return_frames:
                        name = name_fixed + "_" + str(i)
                    else:
                        name_a = name_fixed + r"\A\\" + str(i)
                        name_b = name_fixed + r"\B\\" + str(i)
                    if output_frames_instructions and output_frames_instructions_middle_positioned is False and \
                            output_box is False:
                        cv2.imwrite(name + "_input.jpg", frame)
                    if (
                            position_subtitle_middle is False or output_frames_instructions_middle_positioned) and output_box is False:
                        cv2.putText(frame, text, (int(frame_width * (float(origin_x.strip("%")) / 100)),
                                                  int(frame_height * (float(origin_y.strip("%")) / 100))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
                    elif position_subtitle_middle is False or output_box:
                        # start = (round(int(frame_width * (float(origin_x.strip("%")) / 100))),
                        #         round(int(frame_height * (float(origin_y.strip("%")) / 100))))

                        start = (frame_width - 70,
                                 1)

                        end = (frame_width, 70)
                        # end = (int(frame_width * ((float(origin_x.strip("%")) + float(extent_x.strip("%"))) / 100)),
                        #       int(frame_height * ((float(origin_y.strip("%")) + float(extent_y.strip("%"))) / 100)))

                        # start = (int((start[0] + end[0]) / 2) + 30, int((start[1] + end[1]) / 2) + 30)

                        # end = (start[0] + 70, start[1] + 70)
                        frame = cv2.rectangle(frame, start, end, (0, 255, 0), -1)

                        # cv2.putText(frame, "text", (start[0], int((start[1] + end[1]*3) / 4)),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
                    else:
                        cv2.putText(frame, text, (int(frame_width * 0.375),
                                                  int(frame_height * 0.85) + count_subtitles * int(
                                                      frame_height * 0.04)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

                    if output_frames_instructions_middle_positioned:
                        cv2.putText(frame_middle, text, (int(frame_width * 0.375),
                                                         int(frame_height * 0.85) + count_subtitles * int(
                                                             frame_height * 0.04)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
                    elif output_box:
                        start = (int(frame_width * ((100 - float(extent_x.strip("%"))) / 2) / 100),
                                 int(frame_height * 0.85) + count_subtitles * int(frame_height * 0.04))

                        end = (start[0] + ((float(extent_x.strip("%")) / 100) * frame_width),
                               start[1] + ((float(extent_y.strip("%")) / 100) * frame_height))

                        start = (int((start[0] + end[0]) / 2) + 30, int((start[1] + end[1]) / 2) + 30)

                        end = (start[0] + 70, start[1] + 70)

                        frame_middle = cv2.rectangle(frame_middle, start,
                                                     end,
                                                     (0, 255, 0), -1)

                        # cv2.putText(frame_middle, "text", (start[0], int((start[1] + end[1]*3) / 4)),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))

                count_subtitles = count_subtitles + 1

                if output_frames_instructions is False and return_frames:
                    final_list_frames.append(format_frames(frame, output_size))

                if return_frames is False and output_frames_instructions is False:
                    final_list_input.append(tf.constant([float(len(text)), float(region)]))
                    final_list_target.append(tf.constant([(float(origin_x.strip("%")) / 100),
                                                          (float(origin_y.strip("%")) / 100)]))
                else:
                    if position_subtitle_middle is False or output_frames_instructions_middle_positioned or output_box:
                        if output_frames_instructions is False:
                            final_list_captions.append(tf.constant("Image with positioned subtitle", dtype=tf.string))
                        elif output_frames_instructions_middle_positioned or output_box:
                            # cv2.imwrite(name + "_output_positioned.jpg", frame)
                            # cv2.imwrite(name + "_output_middle.jpg", frame_middle)
                            cv2.imwrite(name_a + ".jpg", frame)
                            cv2.imwrite(name_b + ".jpg", frame_middle)
                        else:
                            cv2.imwrite(name + "_output.jpg", frame)
                            json_dict_array.append({"input": name + "_input.jpg", "output": name + "_output.jpg",
                                                    "request": "Add positioned subtitle " + text})
                            # final_list_captions.append(tf.constant(-1))
                    else:
                        if output_frames_instructions is False:
                            final_list_captions.append(
                                tf.constant("Image with not positioned subtitle", dtype=tf.string))
                        else:
                            cv2.imwrite(name + "_output.jpg", frame)
                            json_dict_array.append({"input": name + "_input.jpg", "output": name + "_output.jpg",
                                                    "request": "Add unpositioned subtitle " + text})
                        # final_list_captions.append(tf.constant(1))
                i += 1
                if safe_training_data and i % 1000 == 0:
                    np.savez_compressed(str(pathlib.Path().resolve().joinpath("dataset_processed", "file_" + str(i))),
                                        np.asarray(final_list_input),
                                        np.asarray(final_list_target))
                    final_list_frames = []
                    if return_frames is False:
                        final_list_target = []
                        final_list_input = []
                    else:
                        final_list_captions = []

                    print("Training batch written")
                if i >= n_frames:
                    break

            if count_subtitles == 0 and return_frames:
                final_list_frames.append(format_frames(frame, output_size))
                final_list_captions.append(tf.constant("Image with no subtitle", dtype=tf.string))
        # elif i < n_frames:
        #    final_list_input.append(np.zeros_like(final_list_input[0]))
        #    final_list_target.append(np.zeros_like(final_list_target[0]))
        #    i += 1
        #    print("empty: " + str(i))
        else:
            break

    # while (i < n_frames):
    #     # while (i < 2000):
    #     ret, frame = cap.read()
    #     if (ret):
    #
    #         # removing black borders on each frame
    #         if crop_video:
    #             frame = crop(frame)
    #             if round_base(frame.shape[0]) != frame_height or round_base(frame.shape[1]) != frame_width:
    #                 frame_height = round_base(frame.shape[0])
    #                 frame_width = round_base(frame.shape[1])
    #
    #         time_frame = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2) + offset
    #         for begin, end, origin_x, origin_y, extent_x, extent_y, text, region in \
    #                 subtitles_array[(subtitles_array[:, 1].astype(float) >= time_frame) &
    #                                 (subtitles_array[:, 0].astype(float) <= time_frame)]:
    #             i = i + 1
    #             final_list_input.append([(frame.astype(np.float16)/255), len(text), region])
    #             final_list_target.append([(float(origin_x.strip("%")) / 100),
    #                                (float(origin_y.strip("%")) / 100)])
    #
    #             if safe_training_data and i % 1000 == 0:
    #                 np.savez_compressed(str(pathlib.Path().resolve().joinpath("data", "file_" + str(i))),
    #                                     np.asarray(final_list_input),
    #                          np.asarray(final_list_target))
    #                 final_list_target = []
    #                 final_list_target = []
    #
    #                 print("Training batch written")
    #         if cv2.waitKey(1) & 0xFF == ord('s'):
    #             break
    #     else:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    #    if export_frame_positions:
    #        import json
    #        with open(name_fixed.replace("_unpositioned", "").replace("_positioned", "") + "frames.json", 'w') as fp:
    #            json.dump(find_faces(final_list_frames_original), fp)
    if len(final_list_frames_original) > 0:
        frame_positions = find_faces(final_list_frames_original, (i - len(final_list_frames_original) + 1),
                                     number_of_times_to_upsample=number_of_times_to_upsampling)
        write_face_annotations(frame_positions,
                               name_fixed.replace("_unpositioned", "").replace("_positioned",
                                                                               "") + "frames.json", i)
    if return_frames is True:
        return start, np.asarray(final_list_frames), np.asarray(final_list_captions)
    elif output_frames_instructions is False:
        return start, np.asarray(final_list_frames), np.asarray(final_list_input), np.asarray(final_list_target)
    elif output_frames_instructions_middle_positioned is False:
        return json_dict_array


def read_in_xml_subtitles(path: str) -> np.ndarray:
    import xml.dom.minidom as md
    # import xml.etree.ElementTree as ET

    # tree = ET.parse(path)
    # root = tree.getroot()

    # for child in root.iter():
    #    print(child.tag, child.attrib)

    doc = md.parse(path)

    subtitles = doc.getElementsByTagName("p")

    # same = False
    begin = -100
    end = -100
    subtitles_array = None
    for subtitle in subtitles:
        attributes = dict(subtitle.attributes.items())
        # begin = subtitle.getAttribute("begin")
        begin = attributes["begin"]
        # end = subtitle.getAttribute("end")
        end = attributes["end"]

        if "t" in begin:
            begin = convert_ticks_in_s(int(begin[:-1]))
            end = convert_ticks_in_s(int(end[:-1]))

        origin = attributes["tts:origin"].split(" ")
        extent = attributes["tts:extent"].split(" ")
        region = int(attributes["region"].strip("region_"))
        # for chi in subtitle.childNodes:
        #    if isinstance(chi, md.Element):
        #        print(chi.firstChild.nodeValue)
        #    else:
        #        print(chi.nodeValue)
        text = " ".join(
            t.firstChild.nodeValue if isinstance(t, md.Element) else t.nodeValue for t in subtitle.childNodes)
        if subtitles_array is None:
            subtitles_array = np.asarray([[begin, end, origin[0], origin[1], extent[0], extent[1], text, region]])
        else:
            subtitles_array = np.append(subtitles_array,
                                        [[begin, end, origin[0], origin[1], extent[0], extent[1], text, region]],
                                        axis=0)
        # print((begin, end, origin, extent, text))

    return subtitles_array
    # print(subtitle.childNodes.nodeValues)
    # print(doc.nodeName)
    # rint(doc.firstChild.tagName)
    # print(items.length)


def convert_ticks_in_s(input: int, tick_rate: int = 10000000):
    assert isinstance(input, int) and isinstance(tick_rate, int), "input is not int"
    return float(round(input / tick_rate, 2))


class FrameGenerator:
    def __init__(self, video_path: str, subtitles_array: np.ndarray, video_offset: float, n_frames: int,
                 frame_step: int = 15, training: bool = False, reuse_start_points: bool = False, batch_size: int = 50,
                 return_frames: bool = True, position_subtitle_middle: bool = False):
        """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
        self.video_path = video_path
        self.video_offset = video_offset
        self.frame_step = frame_step
        self.subtitles_array = subtitles_array
        self.n_frames = n_frames
        self.training = training
        self.reuse_start_points = reuse_start_points
        self.start_points = []
        self.batch_size = batch_size
        self.return_frames = return_frames
        self.position_subtitle_middle = position_subtitle_middle

    # def get_files_and_class_names(self):
    # video_paths = list(self.path.glob('*/*.avi'))
    # classes = [p.parent.name for p in video_paths]
    # return video_paths, classes

    def __call__(self):
        # video_paths, classes = self.get_files_and_class_names()

        # pairs = list(zip(video_paths, classes))
        # for self.runs
        i = 0
        while i <= self.batch_size:
            if self.return_frames is False:
                start, frames, input_data, output_data = create_input_data(video_path=self.video_path, subtitles_array=
                self.subtitles_array,
                                                                           video_offset=self.video_offset,
                                                                           n_frames=self.n_frames,
                                                                           frame_step=self.frame_step,
                                                                           used_start_points=self.start_points,
                                                                           position_subtitle_middle=
                                                                           self.position_subtitle_middle,
                                                                           return_frames=self.return_frames,
                                                                           output_frames_instructions=False)
            else:
                start, frames, captions = create_input_data(video_path=self.video_path, subtitles_array=
                self.subtitles_array,
                                                            video_offset=self.video_offset,
                                                            n_frames=self.n_frames,
                                                            frame_step=self.frame_step,
                                                            used_start_points=self.start_points,
                                                            position_subtitle_middle=
                                                            self.position_subtitle_middle,
                                                            return_frames=self.return_frames,
                                                            output_frames_instructions=False)

            if self.reuse_start_points is False:
                self.start_points.append(start)

            if self.return_frames:
                yield frames, captions
            else:
                yield frames, input_data, output_data

            i = i + 1


class SAM_annotations():
    def __init__(self, path_coco_annotations: list, path_images: str, sam_checkpoint: str
    = r"D:\Gits\SAM\models\sam_vit_h_4b8939.pth", sam_model_type: str = "vit_h", device: str = "cuda", output_path: str
                 = r"D:\Master_Thesis_data\Active_Speaker\pixelmaps"):
        self.final_annotations = None
        self.path_coco_annotations = path_coco_annotations
        self.path_images = path_images
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.device = device

        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        if self.device != "" and self.device is not None:
            self.sam.to(device=self.device)
        else:
            warnings.warn("You do not use CUDA!")
        self.predictor = SamPredictor(self.sam)

        self._read_in_COCO_annotations()
        self.output_path = output_path

        self._create_output_paths()

    def _create_output_paths(self):
        # Get the list of subfolder names in the source directory
        subfolders = [d for d in os.listdir(self.path_images) if os.path.isdir(os.path.join(self.path_images, d))]

        # Check and create subfolders in the destination directory
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.output_path, subfolder)

            if not os.path.exists(subfolder_path):
                # If the subfolder doesn't exist in the destination directory, create it
                os.makedirs(subfolder_path)
                print(f"Created subfolder '{subfolder}' in the destination directory.")
            else:
                print(f"Subfolder '{subfolder}' already exists in the destination directory.")

    def _read_in_COCO_annotations(self):
        self.COCO_info_all = {}
        self.COCO_info = {}
        for path in self.path_coco_annotations:
            with open(path, "r") as file:
                temp_data = json.load(file)

            if r"\train" in path:
                self.COCO_info_all["train"] = temp_data
            elif r"\val" in path:
                self.COCO_info_all["val"] = temp_data
            elif r"test" in path:
                self.COCO_info_all["test"] = temp_data

            temp_images = {pdict["id"]: pdict["file_name"] for pdict in temp_data["images"]}
            for pelement in temp_data["annotations"]:
                self.COCO_info[temp_images[pelement["image_id"]]] = {"bbox": pelement["bbox"], "category_id":
                    pelement["category_id"], "id": pelement["id"], "image_id": pelement["image_id"]}

    def apply_bbox_to_mask(bbox, mask):
        # Extract bounding box coordinates
        x, y, extent_x, extent_y = bbox

        # Create a mask for the bounding box
        bbox_mask = np.zeros(mask.shape, dtype=bool)
        bbox_mask[int(y):int(y) + int(extent_y), int(x):int(x) + int(extent_x)] = True

        # Apply the bounding box mask to the segmentation mask
        mask[~bbox_mask] = False

        return mask

        # self.COCO_info = {}

    def run(self):
        # This is a sample Python script.

        # Press Umschalt+F10 to execute it or replace it with your code.
        # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

        # Press the green button in the gutter to run the script
        state = "train"

        self.final_annotations = {}
        # temp_annotations = {}
        # counter = 0
        for root, directories, files in os.walk(self.path_images):
            # if counter > 5:
            #    break
            if root.split('\\')[-1] == "train":
                state = "train"
            elif root.split('\\')[-1] == "test":
                state = "test"
            elif root.split('\\')[-1] == "val":
                state = "val"
            for filename in files:
                # counter += 1
                # if counter > 5:
                #    break
                file_path = os.path.join(root, filename)
                # Check if the file has a valid image extension (you can add more extensions as needed)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                    # Do something with the image file (e.g., print its path)
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    self.predictor.set_image(image)
                    x, y, x_ext, y_ext = self.COCO_info[filename]["bbox"]
                    input_point = np.array([[x + x_ext / 2, y + y_ext / 2]])
                    input_label = np.array([1])

                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    masks = masks[np.argmax(scores)]
                    logits = logits[np.argmax(scores)]
                    scores = scores[np.argmax(scores)]

                    masks = SAM_annotations.apply_bbox_to_mask(self.COCO_info[filename]["bbox"], masks)
                    masks_temp = masks.copy()
                    # Replace True with 1 and False with 255
                    masks_temp = masks_temp.astype('uint8')
                    unique_values, counts = np.unique(masks_temp, return_counts=True)
                    masks_temp[(masks_temp != 1) & (masks_temp != 0)] = int(255)
                    masks_temp[masks_temp == 1] = int(1)
                    masks_temp[masks_temp == 0] = int(255)
                    unique_values, counts = np.unique(masks_temp, return_counts=True)
                    seg_image = Image.fromarray(masks_temp.astype('uint8'))

                    outputpath = self.output_path + "\\" + state + "\\" + filename.replace(".jpg", ".png")
                    seg_image.save(outputpath)

                    encoded = encode(np.asfortranarray(masks))
                    parea = float(area(encoded))

                    encoded["counts"] = encoded["counts"].decode('utf-8')

                    # decoded = decode(encoded)
                    # decoded[decoded == 0] = int(255)
                    # temp_image = Image.fromarray(decoded)
                    # temp_image.show()
                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(image)
                    # plt.title(f"Mask {1}, Score: {scores:.3f}", fontsize=18)
                    # plt.axis('off')
                    # plt.show()

                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(image)
                    # SAM_annotations.show_points(input_point, input_label, plt.gca())
                    # plt.title(f"Mask {1}, Score: {scores:.3f}", fontsize=18)
                    # plt.axis('off')
                    # plt.show()

                    # plt.figure(figsize=(20, 20))
                    # plt.imshow(image)
                    # SAM_annotations.show_mask(masks, plt.gca())
                    # SAM_annotations.show_points(input_point, input_label, plt.gca())
                    # plt.title(f"Mask {1}, Score: {scores:.3f}", fontsize=18)
                    # plt.axis('off')
                    # plt.show()

                    if state in self.final_annotations:
                        self.final_annotations[state].append(
                            {"segmentation": encoded, "area": parea, "iscrowd": 0, "image_id":
                                self.COCO_info[filename]["image_id"], "bbox": self.COCO_info[filename]["bbox"],
                             "category_id": self.COCO_info[filename]["category_id"],
                             "id": self.COCO_info[filename]["id"]})
                    else:
                        self.final_annotations[state] = [
                            {"segmentation": encoded, "area": parea, "iscrowd": 0, "image_id":
                                self.COCO_info[filename]["image_id"], "bbox": self.COCO_info[filename]["bbox"],
                             "category_id": self.COCO_info[filename]["category_id"],
                             "id": self.COCO_info[filename]["id"]}]

        for pkey, pvalue in self.final_annotations.items():
            temp_output_annot = self.COCO_info_all[pkey]
            temp_output_annot["annotations"] = pvalue

            output_path = self.output_path + "\\" + pkey + "_annotation.json"

            with open(output_path, "w") as file:
                json.dump(temp_output_annot, file)

            # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # plt.axis('on')
        # plt.show()

    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
