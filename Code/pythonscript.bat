D:
cd D:\Shool_WorkSpace\Do_An_Chuan\Code\

rem --------------------------------------------------------------------------------------------DATASET 1--------------------------------------------------------------------------------------------

rem #change top K

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --top_number 9 --seed 8


rem #change negative sample

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --num_train_neg 9 --seed 8

rem --------------------------------------------------------------------------------------------DATASET 2--------------------------------------------------------------------------------------------

rem change top K

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --top_number 9 --seed 8


rem #change negative sample

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_2/ --dataset exp_ml-20m_2 --num_train_neg 9 --seed 8

rem --------------------------------------------------------------------------------------------DATASET 3--------------------------------------------------------------------------------------------

rem change top K

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --top_number 9 --seed 8


rem #change negative sample

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_3/ --dataset exp_ml-20m_3 --num_train_neg 9 --seed 8

rem --------------------------------------------------------------------------------------------DATASET 4--------------------------------------------------------------------------------------------

rem change top K

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --top_number 9 --seed 8


rem #change negative sample

python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/NeuMF_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_PLUS/MLP_PLUS.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/NeuMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/MLP.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 9 --seed 8

python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 1 --seed 0
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 2 --seed 1
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 3 --seed 2
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 4 --seed 3
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 5 --seed 4
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 6 --seed 5
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 7 --seed 6
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 8 --seed 7
python ./Tensorflow/Code_TF_Original/GMF.py --path ../Data/exp_ml-20m_4/ --dataset exp_ml-20m_4 --num_train_neg 9 --seed 8

pause