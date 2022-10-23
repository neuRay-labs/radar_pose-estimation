import argparse, argcomplete
from model_trainer import ModelTrainer

parser = argparse.ArgumentParser(description='radar pose estimation model')
parser.add_argument('--point-cloud-repo-train', help="parent directory containing csv pc files", nargs='+', default=[])
parser.add_argument('--point-cloud-repo-test', help="parent directory containing csv pc files", nargs='+', default=[])
parser.add_argument('--pose-label-repo-train', help="parent directory containing csv pc files", nargs='+', default=[])
parser.add_argument('--pose-label-repo-test', help="parent directory containing csv pc files", nargs='+', default=[])
parser.add_argument("--clearml-task-name", type=str, required=True)
parser.add_argument("--no-validation", action='store_true')
parser.add_argument("--no-clearml", action='store_true')
parser.add_argument("--command-line", action='store_true')
parser.add_argument("--train-presentage", default=80)
parser.add_argument("--batch-size", default=128)
parser.add_argument("--seed", default=1337)
parser.add_argument("--epochs", default=150)
parser.add_argument("--metrics-report-interval", type=int, default=5, help="How many epochs between each metrics report to clearml")
parser.add_argument("--metrics-report-interval", type=float, default = 0.001)


if __name__=="__main__":
    args = parser.parse_args()
    trainer = ModelTrainer(args)
    trainer.train()
