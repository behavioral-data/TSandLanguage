
import argparse
import subprocess
import sys
import time
import os
import errno
import uuid
import re

from pathlib import Path
this_dir = Path(__file__).parents[0]
template_file = this_dir / "slurm-template.sh"

from dotenv import dotenv_values
config = dotenv_values(".env")

def get_number_from_output(output):
    """Get the number from the output of the command using regex."""
    match = re.search(r"(\d+)", output)
    if match:
        return int(match.group(1))
    else:
        return None
    

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

DIR = "${DIR}"
JOB_NAME = "${JOB_NAME}"
JOBS_DIR =   "${JOBS_DIR}"
MEMORY = "${MEMORY}"
ACCOUNT = "${ACCOUNT}"
PARTITION = "${PARTITION}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
NUM_CPUS = "${NUM_CPUS}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
CONDA_PATH = "${CONDA_PATH}"
GIVEN_NODE = "${GIVEN_NODE}"
CONDA_ENV = "${CONDA_ENV}"
LOG_PATH = "${LOG_PATH}"
TIME = "${TIME}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default = "$PWD",
        required=True,
        help="The job's run directory")
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).")
    parser.add_argument(
        "--num-nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to use."),
    parser.add_argument(
        "--account",
        "-a",
        type=str,
        default="bdata",
        help="Account to use")
    parser.add_argument(
        "--memory",
        "-m",
        type=str,
        default="128G",
        help="Account to use")
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default="gpu-rtx6k",
        help="Parition to run on")
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use in each node. (Default: 2)")
    parser.add_argument(
        "--conda-path",
        type=str,
        default="/gscratch/bdata/$USER/anaconda3/bin/conda"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=config["PROJECT_NAME"],
        help="The name of the conda environment")
    
    parser.add_argument(
        "--time",
        type=str,
        default="12:00:00",
        help="The timeout duration for the job. (Default: 12 hours)")
    
    parser.add_argument(
        "--num-cpus",
        type=int,
        required=False,
        default=8,
        help="number of cpus to request per gpu")

    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.")
    parser.add_argument(
        "--dry-run",
        action="store_true")

    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name,
                              time.strftime("%m%d-%H%M", time.localtime()))

    if args.conda_path:
        conda_path_option = "source " +  os.path.join(args.conda_path,"etc","profile.d","conda.sh")
    else:
        conda_path_option = ""

    jobs_dir = os.path.join(this_dir,"jobs")
    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(DIR, args.dir)
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(JOBS_DIR, jobs_dir)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(PARTITION, args.partition)
    text = text.replace(ACCOUNT,args.account)
    text = text.replace(CONDA_PATH,conda_path_option)
    text = text.replace(CONDA_ENV,args.conda_env)
    text = text.replace(MEMORY,args.memory)
    text = text.replace(TIME,args.time)
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(NUM_CPUS, str(args.num_cpus))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")

    # ===== Save the script =====
    script_file_tmp = os.path.join("/usr/tmp","{}.sh".format(uuid.uuid1()))

    with open(script_file_tmp, "w") as f:
        f.write(text)

    if not args.dry_run:
        # ===== Submit the job =====
        print("Starting to submit job!")
        result = subprocess.run(["sbatch", script_file_tmp], stdout=subprocess.PIPE)
        job_id = get_number_from_output(result.stdout.decode("utf-8"))
        
        script_path = os.path.join(jobs_dir,f"{job_name}-{job_id}.sh")
        log_path = os.path.join(jobs_dir,f"{job_name}-{job_id}.out")

        with open(script_path, "w") as f:
            f.write(text)       
            
        print("Job submitted! Script is at {}".format(script_path))
        symlink_force(log_path,"last-slurm.log")
    else:
        print(f"Dry run! Not submitting job. Script file is at {script_file_tmp}")
    
   

    sys.exit(0)

