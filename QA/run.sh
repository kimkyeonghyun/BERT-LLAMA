BS=32
LR=5e-5
EP=10
DATASET=squad

clear
# PADDING='cls'
# MODEL=bert
# LLM=llama3
# METHOD='base'
# LAYER=0
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
# python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
# python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

# MODEL=bert-large
# LLM=llama3
# METHOD='base'
# LAYER=0
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
# python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
# python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=bert
LLM=llama3
METHOD='base_llm'
LAYER=0
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=bert
LLM=llama3
METHOD='base_llm'
LAYER=8
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=bert
LLM=llama3
METHOD='base_llm'
LAYER=16
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=bert
LLM=llama3
METHOD='base_llm'
LAYER=31
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=roberta
LLM=llama3
METHOD='base'
LAYER=0
python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=roberta-large
LLM=llama3
METHOD='base'
LAYER=0
python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}


MODEL=roberta
LLM=llama3
METHOD='base_llm'
LAYER=0
python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=roberta
LLM=llama3
METHOD='base_llm'
LAYER=8
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=roberta
LLM=llama3
METHOD='base_llm'
LAYER=16
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}

MODEL=roberta 
LLM=llama3
METHOD='base_llm'
LAYER=31
# python main.py --task=question_answering --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=question_answering --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
python main.py --task=question_answering --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --llm=${LLM} --method=${METHOD} --layer=${LAYER} --padding=${PADDING}
