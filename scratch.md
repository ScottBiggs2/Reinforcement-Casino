Terrific - now we need to build a new type of job to combine several components: 
* Do dense training (select one or several datasets, dpo or grpo, hyperparams, wandb, save checkpoint to /scratch, good control over number of steps and weight callback logs for later. Make sure the checkpoints and weight callback for mask building go to /scratch)
* Do mask building (just do it all)
* Do Sparse training (do one for each mask type, same datasets/methods/params/logging controls)
* Do evals (original, dense trained, and sparse trained, etc, save results clearly. For speed we can use --limit to keep overhead lower for early checks)

This can be one job, or a job that spins up other jobs. This is a very tall order, but all the independent components exist. 