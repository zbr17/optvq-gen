# training code
WORKSPACE=/data1/code/zbr/optvq-gen WANDB_MODE=offline accelerate launch --gpu_ids 0,1,2,3,4,5 --num_machines=1 --num_processes=6 \
train_maskgit.py config=configs/training/generator/maskgit.yaml \
    experiment.project="optvq_generation" \
    experiment.name="optvq16h4_maskgit" \
    experiment.output_dir="logs/optvq16h4_maskgit"

# evaluation code