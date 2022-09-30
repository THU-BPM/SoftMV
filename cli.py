import argparse
import torch
import os
from typing import Tuple
from prompt.data_loaders import (
    name2datasets,
    UNLABELED_SET,
    TRAIN_SET,
    DEV_SET,
    TEST_SET,
    METRICS,
    DEFAULT_METRICS,
)
from prompt.modules.configs import WrapperConfig
from prompt.utils import set_seed
import prompt.trainers.singleton_trainer as singleton_trainer
import prompt.trainers.configs as promptconfig
import log
import optuna
import optuna.trial as TrialState


logger = log.get_logger("root")


def load_prompt_configs(
    args,
) -> Tuple[
    WrapperConfig, promptconfig.TrainConfig, promptconfig.EvalConfig, promptconfig.DDPConfig
]:
    model_cfg = WrapperConfig(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        wrapper_type=args.wrapper_type,
        task_name=args.task_name,
        label_list=args.label_list,
        max_seq_length=args.prompt_max_seq_length,
        cache_dir=args.cache_dir,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        prompt_length=args.prompt_length,
        prompt_encoder_type=args.prompt_encoder_type,
        init_from_vocab=args.init_from_vocab,
        output_dir=args.output_dir,
        soft_prompt_path=args.soft_prompt_path,
    )
    train_cfg = promptconfig.TrainConfig(
        pattern_lang=args.pattern_lang,
        device=args.device,
        per_gpu_train_batch_size=args.prompt_per_gpu_train_batch_size,
        n_gpu=args.n_gpu,
        num_train_epochs=args.prompt_num_train_epochs,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        cosda_rate=args.cosda_rate,
    )
    eval_cfg = promptconfig.EvalConfig(
        device=args.device,
        n_gpu=args.n_gpu,
        metrics=args.metrics,
        per_gpu_eval_batch_size=args.prompt_per_gpu_eval_batch_size,
    )
    ddp_cfg = promptconfig.DDPConfig(
        do_ddp=args.do_ddp, num_ranks=args.num_ranks, num_nodes=args.num_nodes
    )
    return (model_cfg, train_cfg, eval_cfg, ddp_cfg)


def main(args):

    logger.info("Experiment Parameters: {}".format(args))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )

    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()
    args.task_name = args.task_name.lower()
    if args.task_name not in name2datasets:
        raise ValueError("Task '{}' not found".format(args.task_name))
    data_lang = args.data_lang if args.data_lang else None
    dataset = name2datasets[args.task_name](args.num_shots, data_lang)
    args.label_list = dataset.get_labels()
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)
    prompt_model_cfg, prompt_train_cfg, prompt_eval_cfg, ddp_cfg = load_prompt_configs(args)

    # def objective(trial):
    #     acc = singleton_trainer.train_model_per_pattern(
    #         trial,
    #         model_config=prompt_model_cfg,
    #         train_config=prompt_train_cfg,
    #         eval_config=prompt_eval_cfg,
    #         ddp_config=ddp_cfg,
    #         dataset=dataset,
    #         pattern_ids=args.pattern_ids,
    #         output_dir=args.output_dir,
    #         do_train=args.do_train,
    #         do_eval=args.do_eval,
    #         dict_dir=args.dict_dir,
    #     )
    #     return acc


    # if args.num_shots == 1:
    #     study = optuna.create_study(
    #         study_name="shot1", storage="sqlite:///shot1.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 2:
    #     study = optuna.create_study(
    #         study_name="shot2", storage="sqlite:///shot2.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 4:
    #     study = optuna.create_study(
    #         study_name="shot4", storage="sqlite:///shot4.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 8:
    #     study = optuna.create_study(
    #         study_name="shot8", storage="sqlite:///shot8.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 16:
    #     study = optuna.create_study(
    #         study_name="shot16", storage="sqlite:///shot16.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 32:
    #     study = optuna.create_study(
    #         study_name="shot32", storage="sqlite:///shot32.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 64:
    #     study = optuna.create_study(
    #         study_name="shot64", storage="sqlite:///shot64.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 128:
    #     study = optuna.create_study(
    #         study_name="shot128", storage="sqlite:///shot128.db", direction="maximize", load_if_exists="True"
    #     )
    # elif args.num_shots == 256:
    #     study = optuna.create_study(
    #         study_name="shot256", storage="sqlite:///shot256.db", direction="maximize", load_if_exists="True"
    #     )
    # study.optimize(
    #     objective,
    #     n_trials=100
    # )

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))

    # trial = study.best_trial

    # with open(
    #     os.path.join(args.output_dir, f"{args.num_shots}_params.txt"), "w"
    # ) as fh:
    #     fh.write("Best trial:\n")
    #     fh.write("  Value: {}\n".format(trial.value))
    #     for key, value in trial.params.items():
    #         fh.write("    {}: {}".format(key, value))


    # logger.info("Best trial:")
    # logger.info("  Value: {}".format(trial.value))

    # logger.info("  Params: ")
    # for key, value in trial.params.items():
    #     logger.info("    {}: {}".format(key, value))

    singleton_trainer.train_model_per_pattern(
        prompt_model_cfg,
        prompt_train_cfg,
        prompt_eval_cfg,
        ddp_cfg,
        dataset=dataset,
        pattern_ids=args.pattern_ids,
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        dict_dir=args.dict_dir,
    )


if __name__ == "__main__":
    from parameters import get_args

    args = get_args()
    main(args)