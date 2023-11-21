import click
import subprocess

@click.command()
@click.argument("n_jobs")
@click.argument("sweep_address")
def main(n_jobs, sweep_address):
    command1 = subprocess.Popen(['sbatch',f"--array=1-{n_jobs}",'slurm-utils/slurm_wandb_sweep.sh', sweep_address])

if __name__ == "__main__":
    main()
