import argparse, argcomplete
from email.policy import default
from preprocessor import run_preprocess
import numpy as np
from model_trainer import ModelTrainer
from test_pre_trained import get_prediction, show_plot
from data_set import PreparePoseDataSet

parser = argparse.ArgumentParser(description='Neuray Radar Data Tool')
subparsers = parser.add_subparsers(dest='subcommand')

def subcommand(args=[], parent=subparsers):
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator


def argument(*name_or_flags, **kwargs):
    return ([*name_or_flags], kwargs)

#run training
@subcommand([
# parser = argparse.ArgumentParser(description='radar pose estimation model')
    argument('--point-cloud-repo-train', help="parent directory containing csv pc files", default=""),
    argument('--point-cloud-repo-test', help="parent directory containing csv pc files", default=""),
    argument('--pose-label-repo-train', help="parent directory containing csv pc files", default=""),
    argument('--pose-label-repo-test', help="parent directory containing csv pc files", default=""),
    argument("--output-path", type=str, default='model_runs', help="Where to store all temp models"),
    argument("--clearml-task-name", type=str, required=True),
    argument("--no-validation", action='store_true'),
    argument("--no-clearml", action='store_true'),
    argument("--command-line", action='store_true'),
    argument("--feature-size", default=54),
    argument("--train-percentage", default=80),
    argument("--batch-size", default=128),
    argument("--seed", default=1337),
    argument("--epochs", default=150),
    argument("--metrics-report-interval", type=int, default=15, help="How many epochs between each metrics report to clearml"),
    argument("--learning-rate", type=float, default = 0.001),
])
def train(args):
    trainer = ModelTrainer(args)
    trainer.train()

#preprocess csv pc file into npy
@subcommand([
    argument('--body-pose-path',required = True, help="Path to directory containing pose label needed for Mars preprocessing"),
    argument('--pc-path',required = True, help="Path to directory containing point cloud needed for Mars preprocessing"),
    argument('--train-percentage',default=80, type=int),
    argument('--trail',default=0, type=int),
    argument('--output-path',required = True, help="output path for the npy file. path must contain .npy extention"),
])

def preprocess_data(args):
    dataset = PreparePoseDataSet(args.body_pose_path, args.pc_path, args.trail)
    dataset.save_data(args.train_percentage, args.output_path)
    

#visualize results
@subcommand([
    argument('--freeze-model-path',required = True, help="check point of the model"),
    argument('--npy-test-file',required = True, help='npy file for visualizing the results. MAKE SURE ITS NOT TRAIN FILE'),
    argument('--feature-size', type=int, default = 54)  
])

def visualize_results(args):
    result_test, featuremap_test = get_prediction(args.freeze_model_path, args.npy_test_file, args.feature_size)
    size = result_test.shape[1]//3
    show_plot(featuremap_test, result_test, size)

if __name__ == "__main__":
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)
