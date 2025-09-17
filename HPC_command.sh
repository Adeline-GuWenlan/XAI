#HPC_command.sh
ssh gwl@172.16.51.235
conda activate magic
##############################################################
# ---------------------- 上 传 ----------------------
##############################################################
      # ----------------------语法----------------------
      scp -r /path/to/local_dir username@remote.host:/path/to/remote_dir
         
          #  ----------------------批量----------------------
            # 更新：cab20fd - cab20fd (HEAD -> 7.31whther_attention+Mask) HEAD@{2}: commit: Add experiment_manager.sh
            scp -r /Users/guwenlan/Desktop/XAI/MixMLP/Label\
            gwl@172.16.51.235:/home/gwl/3q/dataset/
            scp -r /Users/guwenlan/Desktop/XAI/Probing/Start_Matching\
            gwl@172.16.51.235:/home/gwl/XAI/Probing
          # ---------------------- 单独文件 ----------------------
            scp Probing/Start_Matching/batch.py\
             gwl@172.16.51.235:/home/gwl/XAI/Probing/Get_Layer_Represen
scp -r /Users/guwenlan/Desktop/XAI/Rawdata/label gwl@172.16.51.235:/home/gwl
##############################################################
# ---------------------- 下 载 ----------------------
##############################################################
scp -r gwl@172.16.51.235:/home/gwl/XAI/visualizations /Users/guwenlan/Desktop

##############################################################
# ----------------------TMUX----------------------
##############################################################

watch -n 0.1 nvidia-smi
