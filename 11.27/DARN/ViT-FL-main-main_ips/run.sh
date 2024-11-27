#!/bin/bash

# 定义要循环的参数值
# 循环嵌套，遍历参数组合
for ds in "News"; do
  for mr in  "0.3" "0.5" "0.7"; do
    for mt in "mcar_" "mnar_p_" "mar_p_"; do
      for np in "0" "149669" "52983"; do

            # 构建命令行参数
            cmd="python train_caFEDAVG.py --dataset $ds --missingrate $mr --missingtype $mt --seed $np "

            # 执行命令
            echo "Running: $cmd"
            $cmd
      done
    done
  done
done
