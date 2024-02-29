# This is a sample Python script.
from dataset.helper.dataset_sampler_helper import __init_arg_parse

import pathlib

# import dlib

if __name__ == '__main__':
    import torch

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the CUDA device count
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s) available.")

        # Get information about each CUDA device
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"Device {i}: {device}, Compute Capability: {capability}")
    else:
        print("CUDA is not available. PyTorch is running on CPU.")

    args = __init_arg_parse()

    if args.task == "subtitle_placement":
        from dataset.helper.dataset_sampler_helper import subtitle_placement

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
