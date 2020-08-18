# Team LHAMa CSCI544 Group Project

## Overview
For our CSCI544 project, we worked on the Stanford Question Answering [0] task, specifically the 2.0 version that included unanswerable questions written in an adversarial way. We were interested in defining new ways to create unanswerable, adversarial questions and perform the question answering task using different architectures. To aid with the latter goal, we heavily leveraged the HuggingFace [1] and we crafted two methods quickly write seemingly-answerable-but-really-unanswerable questions.

The HuggingFace code was cloned into the `transformers` folder of this repo and the following main changes were made:
- Create new file to hold our custom models in `transformers/transformers/modeling_bert_lhama_extensions.py`
- Update SQuAD runner to support our models in `transformers/examples/run_squad.py`
- Support imports of our new models in `transformers/transformers/__init__.py`

We created nearly 200 new unanswerable questions with our two methods and added them to the `transformers/input/train-v2.1-full.json` file. Training of our models was done on both the SQuAD v2.0 and our new v2.1 datasets on Google Cloud Deep Learning VMs. Raw questions can be found here: https://docs.google.com/spreadsheets/d/1U_hFs7XXiaMlVAUdkKGQwgfIHjilrFB_SVV0OuLtk9s/.

More details can be found in our full report.

## Requirements to Run
- Tested only on MacOS/Linux
- Python 3.7

## Installation on Laptop/VM
- Download the code locally:        `git clone https://github.com/USC-LHAMa/CSCI544_Project`
- Navigate to the code directory:   `cd CSCI544_Project`
- Run setup script:                 `sh cloud_vm_setup.sh`
- (optional) tmux for MacOS:        `brew install tmux`

## Run SQuAD Task
- Activate virtual environment:             `source ./bin/activate`
- Navigate to the transformers directory:   `cd transformers`
- The command below runs SQuAD training and evaluation using the LHAMa CNN model on the augmented LHAMa SQuAD 2.1 dataset for 4 epochs and evaluates the model against the SQuAD 2.0 dev set.
- Valid values for `--model_type`:
  - `bert`: for vanilla BERT provided by Google/HuggingFace
  - `lhamalinear`: for LHAMa extended linear model
  - `lhamalstm`: for LHAMa LSTM model
  - `lhamacnn`: for LHAMa CNN model
- Valid values for `--train_file`:
  - `input/train-v2.0.json`: SQuAD 2.0 training set
  - `input/train-v2.1.json`: LHAMa SQuAD 2.1 training set with additional adversarial questions
- To reproduce our results, run eight combinations of the four model types with the two training files

`python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py --model_type lhamacnn --model_name_or_path bert-base-uncased --do_train --do_lower_case --do_eval --train_file input/train-v2.1.json --predict_file input/dev-v2.0.json --learning_rate 2e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 --output_dir ../models/lhamacnn/ --per_gpu_eval_batch_size=12 --per_gpu_train_batch_size=12 --version_2_with_negative --local_rank=-1 --overwrite_output_dir --save_steps 5000`

For these types of long-running tasks, it is best to use tmux on the VM.
- Open a terminal on your laptop, e.g. default MacOS terminal or iTerm2
- SSH into the VM with your account: `gcloud compute ssh <user_name>@<vm_name>`
- Start tmux on the VM: `tmux`
- Navigate to the `transformers` directory and run the command to run the SQuAD task
- If you are disconnected, SSH back into the server and attach to the previous session: `tmux attach-session -t <session_id>`
  - Existing tmux sessions can be viewed with `tmux ls`
  - If you connect to the wrong session: `tmux detach-client -s <attached_session_name>`

## References
- [0] - SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- [1] - HuggingFace Transformers: https://github.com/huggingface/transformers