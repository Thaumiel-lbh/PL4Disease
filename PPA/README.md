# PPA CODE
## Quick Start
1. git clone our respository.
```
git clone https://github.com/Thaumiel-lbh/PL4Disease.git
```
2. cd PPA code workspace
```
cd PPA/code
```
3. run scripts
order_1:
```
python main_ours_order1.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 10-31-modify/order1-ours-003 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
``` 
order_2:
```
python main_ours_order2.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
```
order_3
```
python main_ours_order3.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 110 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order3-ours-001 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
```
## Train with your own dataset
1. check "ppa-order*-std.xls" in PPA code dir, modify it with your own datasets.
2. check utils.py in PPA code dir, run train scripts with modified parameter ```--data_root```.
