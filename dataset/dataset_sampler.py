# This is a sample Python script.
from helper import create_input_data
import argparse
import pathlib
import dlib

if __name__ == '__main__':

    # ToDo: transfer so separate options module
    parser = argparse.ArgumentParser(description="Provide input parameters")

    parser.add_argument("--crop", dest="crop_video", action="store_true", default=False,
                        help='define if black borders should be removed from Video (longer runtime)')
    parser.add_argument("--vo", dest="video_offset", default=None, action="store", type=float,
                        help='define offset hardcoded. Please provide either --vo or --vd')
    parser.add_argument("--vd", dest="video_duration", default=None, action="store", type=float,
                        help='define offset hardcoded. Please provide either --vo or --vd')
    parser.add_argument("--vp", dest="video_path", default=str(pathlib.Path.home()) + '/OneDrive - '
                                                                                      'IUBH Internationale Hochschule/'
                                                                                      'Vorlesungen/Master Thesis/'
                                                                                      'Video Recordings/'
                                                                                      'Netflix - SWAT - train.mp4',
                        action="store", type=str,
                        help='define offset hardcoded. Please provide either --vo or --vd')
    parser.add_argument("--bs", dest="batch_size", default=1, action="store", type=int, help='define batch size')
    parser.add_argument("--rf", dest="return_frames", default=False, action="store_true", help='defines if frames with '
                                                                                               'subtitles or frames '
                                                                                               'with separate target '
                                                                                               'position vector is '
                                                                                               'returned')
    parser.add_argument("--nf", dest="n_frames", default=200, action="store", type=int, help='define number of frames')
    parser.add_argument("--box", dest="output_box", default=False, action="store_true", help='defines if frames with '
                                                                                             'boxes (black) for '
                                                                                             'subtitle position are '
                                                                                             'displayed instead of '
                                                                                             'subtitles')
    parser.add_argument("--td", dest="train_data", default=False, action="store_true", help='defines if training or '
                                                                                            'test dataset should be '
                                                                                            'created')
    parser.add_argument("--tv", dest="validation_data", default=False, action="store_true", help='defines if training or '
                                                                                            'test dataset should be '
                                                                                            'created')
    parser.add_argument("--rc", dest="return_captions", default=False, action="store_true",
                        help='defines if captions json file should be provided')
    parser.add_argument("--riwa", dest="remove_images_without_annotations", default=False, action="store_true",
                        help='defines if images in json files should only contain image references having any annotation '
                             '(or caption)')
    parser.add_argument("--stp", dest="subtitles_path", default=str(pathlib.Path().resolve().
                                                                                           joinpath("subtitles",
                                                                                                    "netflix_sample.xml"
                                                                                                    )) ,
                        action="store", type=str, help="Provide path to subtitles")
    parser.add_argument("--frc", dest="fractions", default=[0.75, 0.15, 0.1], type=float, nargs="+",
                        help="List of values to define fractions of train, val and test")
    parser.add_argument("--tsk", dest="task", default="subtitle_placement", type=str, action="store")
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
    parser.add_argument("--overlay_frames", dest="overlay_frames", type=int, default=1)
    parser.add_argument("--overlay_frames_skip", dest="overlay_frames_skip", type=int, default=1)
    args = parser.parse_args()

    if args.task == "subtitle_placement":
        from helper import subtitle_placement
        subtitle_placement = subtitle_placement(video_path=args.video_path, subtitles_path=args.subtitles_path,
                                                task_name=args.task_name, video_offset=args.video_offset,
                                                n_frames=args.n_frames, video_duration=args.video_duration,
                                                frame_step=args.frame_step, fractions=args.fractions,
                                                default_middle=True,
                                                extract_middle_and_default=args.extract_middle_and_default,
                                                dot_middle_of_subtitle_box=args.dot_middle_of_subtitle_box,
                                                fixed_start_point=args.fixed_start_point,
                                                extract_annotations=args.extract_annotations,
                                                overlay_frames_skip=args.overlay_frames_skip,
                                                overlay_frames=args.overlay_frames, frames_per_step=args.frames_per_step)
        subtitle_placement.create_input_data()
    elif args.task == "SAM":
        from helper import SAM_annotations
        SAM_segment_creator = SAM_annotations(path_coco_annotations=[r"D:\Master_Thesis_data\Active_Speaker\data\train"
                                                                     r"\result.json",
                                               r"D:\Master_Thesis_data\Active_Speaker\data\val\result.json",
                                               r"D:\Master_Thesis_data\Active_Speaker\data\test\result.json"],
                                              path_images=r"D:\Master_Thesis_data\Active_Speaker\data",
                                              output_path=r"D:\Master_Thesis_data\Active_Speaker\pixelmaps")
        SAM_segment_creator.run()
    else:
        pass
