import argparse
import copy
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


############### Libero Imports ###############
current_working_directory = os.getcwd()
os.chdir(os.environ['PYTHONPATH'])
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, task_orders
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.metric import (
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    safe_device,
    torch_load_model,
)
from libero.lifelong.main import get_task_embs
import robomimic.utils.obs_utils as ObsUtils
from libero.lifelong.algos import get_algo_class
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.chdir(current_working_directory)


############### Lerobot Imports ###############
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.policies.factory import make_policy
from pathlib import Path



from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

def load_policy(pretrained_policy_path):
    pretrained_policy_path = Path(pretrained_policy_path)
    hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), None)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    policy.eval()

    cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), None)
    return policy, cfg


def get_action(cfg, model, obs, task_label, processor=None):
    model = get_model(cfg)
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action




def observation_to_desired_shape(observation):
    desired_observation = {}

    # Extract gripper and joint states and concatenate them
    gripper_states = observation['obs']['gripper_states']
    joint_states = observation['obs']['joint_states']
    state = torch.cat((gripper_states, joint_states), dim=1)  # Shape [20, 9]

    # Extract image observation and ensure desired shape
    agentview_rgb = observation['obs']["agentview_rgb"]  # Shape [20, 3, 128, 128]
    agentview_rgb = agentview_rgb.flip(-1).flip(-2)

    # Set the stacked tensors to the desired_observation dictionary
    desired_observation['observation.state'] = state
    desired_observation['observation.images.image'] = agentview_rgb

    return desired_observation



# TODO: check whether the algo is created correctly
algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model_path_folder", type=str, default="/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all/")
    parser.add_argument("--pretrained_policy_path", type=str,
                        default="/mnt/arc/yygx/pkgs_baselines/lerobot/outputs/train/2024-11-21/16-45-45_pusht_diffusion_default/checkpoints/005000/pretrained_model")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal", "yy_try",
                 "modified_libero", "single_step"],
        default="libero_90"
    )
    parser.add_argument("--folder_name", type=str, default="act_4")
    parser.add_argument("--task_order_index", type=int, default=5)
    parser.add_argument("--task_num_to_use", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True, default=10000)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    return args

def main():
    args = parse_args()
    
    # Get the benchmarks
    benchmark = get_benchmark(args.benchmark)(args.task_order_index, n_tasks_=args.task_num_to_use)
    n_tasks = benchmark.n_tasks
    task_id_ls = task_orders[args.task_order_index]

    # Obtain language descriptions
    descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
    print("======= Tasks Language =======")
    print(f"{descriptions}")
    print("======= Tasks Language =======")

    succ_list = []
    eval_task_id = []
    # yy: for task_idx in range(n_tasks): will make args.task_num_to_use meaningless and lead to wrong task_idx
    # for task_idx in range(n_tasks):
    for task_idx, task_id in enumerate(task_id_ls):  # task_id is the actual id of the task. task_idx is just the index.
        print(f">> Evaluate on original Task {task_id}")
        # Obtain useful info from saved model - checkpoints / cfg
        model_index = task_id
        model_path = args.model_path_folder
        model_path = os.path.join(model_path, f"task{model_index}_model.pth")
        if not os.path.exists(model_path):
            print(f">> {model_path} does NOT exist!")
            print(f">> Env_{task_id} evaluation fails.")
            continue
        # TODO
        _, cfg, previous_mask = torch_load_model(
            model_path, map_location=args.device_id
        )

        # Modify some attributes of cfg via args
        cfg.benchmark_name = args.benchmark
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")
        cfg.device = args.device_id
        save_dir = os.path.join("outputs/libero_eval", f"eval_tasks_on_ori_envs_diffusion_{args.folder_name}_seed{args.seed}", f"evaluation_task{task_id}_on_ori_envs")
        print(f">> Create folder {save_dir}")
        os.system(f"mkdir -p {save_dir}")

        # Create algo
        # TODO
        # algo = safe_device(get_algo_class(algo_map["base"])(n_tasks, cfg), cfg.device)
        # algo.policy.load_state_dict(sd)
        policy, cfg_diffusion = load_policy(pretrained_policy_path=args.pretrained_policy_path)



        # Obtain language embs & task
        task_embs = get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)
        task = benchmark.get_task(task_idx)

    
        """
        Start Evaluation
        """
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
        test_loss = 0.0

        save_stats_pth = os.path.join(
            save_dir,
            f"load_ori_{model_index}_on_ori_{task_id}.stats",
        )
    
        video_folder = os.path.join(
            save_dir,
            f"load_ori_{model_index}_on_ori_{task_id}_videos",
        )

        os.system(f"mkdir -p {video_folder}")


        with Timer() as t:
            video_writer_agentview = VideoWriter(os.path.join(video_folder, "agentview"), save_video=True,
                                                 single_video=False)
            video_writer_wristcameraview = VideoWriter(os.path.join(video_folder, "wristcameraview"), save_video=True,
                                                       single_video=False)

            env_args = {
                "bddl_file_name": os.path.join(
                    cfg.bddl_folder, task.problem_folder, task.bddl_file
                ),
                "camera_heights": cfg.data.img_h,
                "camera_widths": cfg.data.img_w,
            }

            env_num = cfg['eval']['n_eval']
            env = SubprocVectorEnv(
                [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
            )
            env.reset()
            env.seed(cfg.seed)
            init_states_path = os.path.join(
                cfg.init_states_folder, task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            indices = np.arange(env_num) % init_states.shape[0]
            init_states_ = init_states[indices]
    
            dones = [False] * env_num
            steps = 0
            obs = env.set_init_state(init_states_)
            task_emb = benchmark.get_task_emb(task_idx)
    
            num_success = 0
            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros((env_num, 7)))

            prev_observation = None

            with torch.no_grad():
                while steps < cfg.eval.max_steps:
                    print(f">> step: {steps}")
                    steps += 1
                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                    # TODO
                    # data = observation_to_desired_shape(data, prev_observation=prev_observation, n_obs_steps=cfg_diffusion.policy.n_obs_steps)
                    data = observation_to_desired_shape(data)
                    # prev_observation = copy.deepcopy(data)
                    with torch.inference_mode():
                        actions = policy.select_action(data)
                    #     TODO: type of action has some issues here
                    actions = actions.cpu().tolist()
                    obs, reward, done, info = env.step(actions)
                    video_writer_agentview.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )
                    video_writer_wristcameraview.append_vector_obs(
                        obs, dones, camera_name="robot0_eye_in_hand_image"
                    )
                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    if all(dones):
                        break
                for k in range(env_num):
                    num_success += int(dones[k])

            video_writer_agentview.save(save_video_name="video_agentview")
            video_writer_wristcameraview.save(save_video_name="video_wristcameraview")
            success_rate = num_success / env_num
            env.close()
    
            eval_stats = {
                "loss": test_loss,
                "success_rate": success_rate,
            }

            succ_list.append(success_rate)
            torch.save(eval_stats, save_stats_pth)
            with open(os.path.join("outputs/libero_eval", f"eval_tasks_on_ori_envs_diffusion_{args.folder_name}_seed{args.seed}", f"succ_list_evaluation_on_ori_envs.npy"), 'wb') as f:
                np.save(f, np.array(succ_list))

        with open(os.path.join("outputs/libero_eval", f"eval_tasks_on_ori_envs_diffusion_{args.folder_name}_seed{args.seed}", f"succ_list_evaluation_on_ori_envs.npy"), 'wb') as f:
            np.save(f, np.array(succ_list))
        print(
            f"[info] finish for ckpt at {model_path} in {t.get_elapsed_time()} sec for rollouts"
        )
        print(f"Results are saved at {save_stats_pth}")
        print(test_loss, success_rate)
        eval_task_id.append(task_id)

    print(f"[INFO] Finish evaluating original env list: {eval_task_id}")

if __name__ == "__main__":
    main()

