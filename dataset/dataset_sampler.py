# This is a sample Python script.
from dataset.helper.dataset_sampler_helper import __init_arg_parse

import pathlib

# import dlib

if __name__ == '__main__':
    import torch, dlib

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the CUDA device count
        device_count = torch.cuda.device_count()
        print(f"Torch: Found {device_count} CUDA device(s) available.")

        # Get information about each CUDA device
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"Torch: Device {i}: {device}, Compute Capability: {capability}")
    else:
        print("Torch: CUDA is not available. PyTorch is running on CPU.")

    if dlib.DLIB_USE_CUDA:
        print("dlib: CUDA support is enabled.")
    else:
        print("dlib: CUDA support is not enabled.")

    args = __init_arg_parse()

    if args.task == "dataset_generation_1":
        from dataset.helper.dataset_sampler_helper import SubtitlePlacement as subtitle_placement

        subtitle_placement._check_args_init(args)
        subtitle_placement = subtitle_placement(video_path=args.video_path, subtitles_path=args.subtitles_path,
                                                task_name=args.task_name, video_offset=float(args.video_offset),
                                                n_frames=args.n_frames, video_duration=args.video_duration,
                                                frame_step=args.frame_step, fractions=args.fractions,
                                                default_middle=True,
                                                extract_middle_and_default=args.extract_middle_and_default,
                                                dot_middle_of_subtitle_box=args.dot_middle_of_subtitle_box,
                                                fixed_start_point=args.fixed_start_point,
                                                extract_annotations=args.extract_annotations,
                                                overlay_frames_skip=args.overlay_frames_skip,
                                                overlay_frames=args.overlay_frames,
                                                frames_per_step=args.frames_per_step)
        subtitle_placement.create_input_data()
    elif args.task == "dataset_generation_2":

        def filter_dataset(path_jsons, path_dataset, output_path, rebalance:bool = True):
            import os
            import json
            # Create a dictionary to store data based on folder names
            data_dict = {}

            # Iterate over json files in path_jsons
            for root, dirs, files in os.walk(path_jsons):
                base_folder = root.replace(path_jsons, "").replace(os.path.basename(root), "").strip("\\")
                for file in files:
                    if file.endswith('.json'):
                        folder_name = os.path.basename(root)
                        with open(os.path.join(root, file), 'r') as json_file:
                            data = json.load(json_file)
                            data = ["-".join(el["file_name"].split("-")[1:]) for el in data["images"]]
                            if base_folder not in data_dict:
                                data_dict[base_folder] = {}

                            if folder_name not in data_dict[base_folder]:
                                data_dict[base_folder][folder_name] = []
                            data_dict[base_folder][folder_name].extend(data)
            if rebalance:
                raise NotImplementedError("OhOh")

            # Iterate over subfolders in path_dataset
            assert any("single" in folder_name for folder_name in os.listdir(path_dataset)) and \
                   any("overlapped" in folder_name for folder_name in os.listdir(path_dataset)) and \
                   any("voting" in folder_name for folder_name in os.listdir(path_dataset)), (
                "Please follow the instructions in the README.md file. "
                "Provide three separate folders for single, overlapping, and voting experiment."
            )

            for root, dirs, files in os.walk(os.path.join(path_dataset, [folder_name for folder_name in
                                                                         os.listdir(path_dataset) if "single" in
                                                                                                     folder_name][0])):

                if root.split("\\")[-2] == "\\A" or root.split("\\")[-2] == "B" or root.split("\\")[-2] == "_A":
                    for file in files:
                        if root.split("\\")[-2] == "A":
                            pass
                        elif root.split("\\")[-2] == "B":
                            if "-".join(file.split("-")[:1]) in data_dict[[folder_name for folder_name in
                                                                         os.listdir(path_dataset) if "single" in
                                                                                                     folder_name][0]]:
                                pass
                        else:
                            pass

                        if dirs in data_dict:
                            # Create folder in output_path if it doesn't exist
                            output_folder = os.path.join(output_path, folder_name)
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

                            # Iterate over files in the subfolder
                            for file_name in os.listdir(os.path.join(path_dataset, folder_name)):
                                # Perform filtering based on data in data_dict
                                filtered_data = [data for data in data_dict[folder_name] if data[
                                    'filter_criteria'] == file_name]  # Modify the filter_criteria as per your data
                                # Write filtered data to output_path
                                with open(os.path.join(output_folder, file_name), 'w') as output_file:
                                    json.dump(filtered_data, output_file)


        # Example usage:
        path_jsons = r'G:\Coding\VLM_subtitling\dataset_labeled'
        path_dataset = 'G:\Coding\VLM_subtitling\dataset_processed'
        output_path = "\\".join(args.video_path.split("\\")[:-1]) + r"\dataset_final"
        filter_dataset(path_jsons, path_dataset, output_path)

        exit(-1)
        path_coco_annotations = args.annotations
        path_images = args.video_path
        output_path = "\\".join(args.video_path.split("\\")[:-1]) + r"\dataset_final"
    elif args.task == "SAM":
        from dataset.helper.dataset_sampler_helper import SAM_annotations

        assert args.annotations
        assert not "." in args.video_path and not ".mp4" in args.video_path, ("Please provide valid image path for "
                                                                              "SAM segmentation task, instead of "
                                                                              "video file")

        SAM_segment_creator = SAM_annotations(path_coco_annotations=args.annotations,
                                              path_images=args.video_path,
                                              output_path=args.video_path + r"\active_speaker_pixelmaps")
        SAM_segment_creator.run()
    else:
        pass
