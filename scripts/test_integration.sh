#!/bin/bash
set -e

# get HF_TOKEN
. ~/.rc

set -x
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.1
RUNDIR=/opt/sdg-tests

function ensureDependencies(){
  # TODO: install  cuda drivers
  sudo dnf install -y podman 
}

function ensureVLLM(){
   # Ensure vllm is running
   if [ $(podman inspect --format  '{{.State.Running}}' vllm) != "true" ] ; then
      podman run --name vllm -d --device nvidia.com/gpu=all -e HF_TOKEN --replace -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 docker.io/vllm/vllm-openai:latest --model $MODEL_ID
   fi
    while ! curl http://localhost:8000/v1/models -H "Content-Type: application/json" ; do sleep 5 ; done
}


function installEnv(){
    sudo mkdir -p $RUNDIR
    sudo chown $(id -u) $RUNDIR


    if [ ! -e $RUNDIR/Research-sdg ] ; then
        git clone https://github.com/Red-Hat-AI-Innovation-Team/SDG-Research.git $RUNDIR/Research-sdg
    fi
    
    cd $RUNDIR/Research-sdg
    
    # TODO: git reset to the relevant branch/sha
    
    python -m venv venv
    . venv/bin/activate
    
    pip install .
}


function runTest(){
    mkdir -p $RUNDIR/integrationtest
   
    # TODO: these files would be in the repository somewhere
    cat - <<-EOF > $RUNDIR/integrationtest/simple.yaml
- block_type: LLMBlock
  block_config:
    block_name: gen_skill_freeform
    config_path: $RUNDIR/integrationtest/simple_config.yaml
    model_id: $MODEL_ID
    output_cols:
      - output
  gen_kwargs:
    temperature: 0
    max_tokens: 5
  drop_duplicates:
    - output
EOF

    cat - <<-EOF > $RUNDIR/integrationtest/simple_config.yaml
system: You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.
introduction: Develop a series of question and answer pairs to perform a task.
principles: |
  1. You should keep your response brief
  2. Follow the example format
  3. Do not copy the example question and answer pair
examples: |
  The task is {{task_description}}.
  Here is an example to help you understand the type of questions that are asked for:

  Q. {{seed_question}}
  A. {{seed_response}}

generation: |
  Provide a single question and answer pair based on the examples.
start_tags: [""]
end_tags: [""]
EOF

    cat - <<-EOF > $RUNDIR/integrationtest/simple_seed_data.yaml
{"seed_response":"ocean","seed_question":"A large body of water between continents", "task_description":"Your task is to create a description of a object and a single word nameing the object","simple_task_description":"Create a description of a object and a single word nameing the object"}
EOF


    cd $RUNDIR/Research-sdg
    python run.py --ds_path $RUNDIR/integrationtest/simple_seed_data.yaml  --save_path /tmp/output.jsonl --checkpoint_dir $RUNDIR/checkpoints --endpoint http://localhost:8000/v1 --flow $RUNDIR/integrationtest/simple.yaml 

    # The output isn't guarantee deterministic but with temperature==0
    # and a very short max_tokens the output should be more likely to be consistent
    [ "$(cat /tmp/output.jsonl | jq .output)" == '"Q. A large,"' ]

    echo "DONE"

}

ensureDependencies
ensureVLLM
installEnv
runTest
