import argparse

class Options():
    def initialize(self, parser):
        parser.add_argument('--experiment_name', default='my_experiment', help='the name of the experiment')
        parser.add_argument('--seed', type=int, default=3407)
        parser.add_argument('--num_workers', type=int, default=4)

        # Multi-GPU training
        parser.add_argument('--use_multi_gpu', action='store_true',
                            help='Use DataParallel for multi-GPU training')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                            help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')

        parser.add_argument('--train_data_root', default='')
        parser.add_argument('--train_classes', nargs='+', default=['car', 'cat', 'chair', 'horse'],
                            help='The image categories included in the training set')
        parser.add_argument('--val_data_root', default='')
        parser.add_argument('--val_classes', nargs='+', default=['car', 'cat', 'chair', 'horse'],
                            help='The image categories included in the validation set')

        # JSON dataset paths (for custom dataset)
        parser.add_argument('--use_json_dataset', action='store_true',
                            help='Use JSON-based dataset loading instead of ImageFolder')
        parser.add_argument('--train_json', type=str,
                            default='/data/data/deepfake_finetune_dataset/jsons/train.json',
                            help='Path to training JSON file')
        parser.add_argument('--val_json', type=str,
                            default='/data/data/deepfake_finetune_dataset/jsons/val.json',
                            help='Path to validation JSON file')
        parser.add_argument('--test_json', type=str,
                            default='/data/data/deepfake_finetune_dataset/jsons/test.json',
                            help='Path to test JSON file')

        # Stage control
        parser.add_argument('--training_stage', type=int, default=2,
                            help='Training stage: 1 or 2')

        # Stage 1 settings
        parser.add_argument('--stage1_batch_size', type=int, default=32)
        parser.add_argument('--stage1_epochs', type=int, default=50)
        parser.add_argument('--stage1_learning_rate', type=float, default=5e-5)
        parser.add_argument('--stage1_lr_decay_step', type=int, default=2)
        parser.add_argument('--stage1_lr_decay_factor', type=float, default=0.7)
        parser.add_argument('--WSGM_count', type=int, default=12)
        parser.add_argument('--WSGM_reduction_factor', type=int, default=4)

        # Pretrained model loading for Stage 1
        parser.add_argument('--pretrained_stage1_path', type=str, default='',
                            help='Path to pretrained Stage 1 model for fine-tuning (e.g., ForgeLens pretrained weights)')

        # Intermediate model path
        parser.add_argument('--intermediate_model_path', default='',
                            help='Path to the intermediate model saved after stage 1')

        # Stage 2 settings
        parser.add_argument('--stage2_batch_size', type=int, default=16)
        parser.add_argument('--stage2_epochs', type=int, default=10)
        parser.add_argument('--stage2_learning_rate', type=float, default=2e-6)
        parser.add_argument('--stage2_lr_decay_step', type=int, default=2)
        parser.add_argument('--stage2_lr_decay_factor', type=float, default=0.7)
        parser.add_argument('--FAFormer_layers', type=int, default=2)
        parser.add_argument('--FAFormer_reduction_factor', type=int, default=1)
        parser.add_argument('--FAFormer_head', type=int, default=2)

        # evaluation setting
        parser.add_argument('--eval_stage', type=int, default=1,
                            help='which stage do you want to evaluate, stage: 1 or 2?')
        parser.add_argument('--weights', default='', help='trained model weights')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--eval_data_root', default='')

        # inference setting
        parser.add_argument('--input_dir', default='')
        parser.add_argument('--output_dir', default='')

        # Data preprocessing mode
        parser.add_argument('--use_resize_only', action='store_true',
                            help='Use resize (nearest neighbor) instead of crop for data preprocessing. '
                                 'When enabled, images are resized to 224x224 with nearest neighbor interpolation. '
                                 'When disabled (default), images use random crop (training) or center crop (evaluation).')

        return parser

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.opt = parser.parse_args()

        if not self.opt.intermediate_model_path:
            if self.opt.experiment_name:
                self.opt.intermediate_model_path = f"./check_points/{self.opt.experiment_name}/train_stage_1/model/intermediate_model_best.pth"
            else:
                raise ValueError("experiment_name must be specified if intermediate_model_path is not provided.")

        if not self.opt.weights:
            if self.opt.experiment_name:
                if self.opt.eval_stage == 1:
                    self.opt.weights = f"./check_points/{self.opt.experiment_name}/train_stage_1/model/intermediate_model_best.pth"
                else:
                    self.opt.weights = f"./check_points/{self.opt.experiment_name}/train_stage_2/model/model_best_val_loss.pth"
            else:
                raise ValueError("experiment_name must be specified if model weights is not provided.")

        return self.opt


    def print_options(self):
        print("----------- Configuration Options -----------")
        for k, v in sorted(vars(self.opt).items()):
            print(f"{k}: {v}")
        print("---------------------------------------------")
