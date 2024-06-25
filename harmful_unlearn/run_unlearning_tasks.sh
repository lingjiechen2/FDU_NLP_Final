#!/bin/bash

# Run OPT Models
srun -p a800 --gres=gpu:3 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 2e-4 --max_unlearn_steps 1000 --model_name=facebook/opt-2.7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/OPT_step_1000_batch_2 --batch_size 2 --log_file=logs/opt-2.7b-unlearn_step_1000_batch_2.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/OPT_step_1000_batch_8 --new_data_formulation

srun -p a800 --gres=gpu:4 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 2e-4 --max_unlearn_steps 500 --model_name=facebook/opt-2.7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/OPT_step_500_batch_8 --batch_size 8 --log_file=logs/opt-2.7b-unlearn_step_500_batch_8.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/OPT_step_500_batch_8 --new_data_formulation --save_interval 50


srun -p a800 --gres=gpu:3 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 2e-4 --max_unlearn_steps 100 --model_name=facebook/opt-2.7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/OPT_step_1000_batch_8 --batch_size 8 --log_file=logs/opt-2.7b-unlearn_step_1000_batch_8.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/OPT_step_1000_batch_8 --new_data_formulation


# Run LLAMA-7B
srun -p a800 --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 2e-4 --max_unlearn_steps 500 --model_name=meta-llama/Llama-2-7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/LLAMA_7B_step_500_batch_2_lr_2e-5 --batch_size 2 --log_file=logs/llama-7b-unlearn_step_500_batch_2_lr_2e-5.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/LLAMA_7B_step_500_batch_2_lr_2e-5

srun -p a800 --gres=gpu:4 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 2e-5 --max_unlearn_steps 1000 --model_name=meta-llama/Llama-2-7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/LLAMA_7B_step_1000_batch_2_lr_2e-5 --batch_size 2 --log_file=logs/llama-7b-unlearn_step_1000_batch_2_lr_2e-5.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/LLAMA_7B_step_1000_batch_2_lr_2e-5

srun -p a800 --gres=gpu:4 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 6e-5 --max_unlearn_steps 1000 --model_name=meta-llama/Llama-2-7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/LLAMA_7B_step_1000_batch_2_lr_6e-5 --batch_size 2 --log_file=logs/llama-7b-unlearn_step_1000_batch_2_lr_6e-5.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/LLAMA_7B_step_1000_batch_2_lr_6e-5

srun -p a800 --gres=gpu:4 --cpus-per-task=12 --mem-per-cpu=8G --pty python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 --lr 2e-4 --max_unlearn_steps 1000 --model_name=meta-llama/Llama-2-7b --model_save_dir=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/LLAMA_7B_step_1000_batch_2_lr_2e-4 --batch_size 2 --log_file=logs/llama-7b-unlearn_step_1000_batch_2_lr_2e-4.log --task_vector_saving_path=/remote-home/miintern1/SKU/harmful_unlearn/trained_models/task_vector/LLAMA_7B_step_1000_batch_2_lr_2e-4
