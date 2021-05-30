result_path=../results/PoST
data_path=../data/PoST

for exp in ${result_path}/*
do 
    echo evaluating ${exp}...
    python eval.py --result_path ${exp} --data_path ${data_path} --threshs 0.16 0.08 0.04
done