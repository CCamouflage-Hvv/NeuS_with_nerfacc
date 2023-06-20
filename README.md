# NeuS_with_nerfacc
 ~
nerfacc的rendering函数需要改动： 
在.conda\envs\NeuS_with_nerfacc\Lib\site-packages\nerfacc\volrend.py, line 116处：
改为rgbs, alphas,network_output = rgb_alpha_fn(t_starts, t_ends, ray_indices)

在.conda\envs\NeuS_with_nerfacc\Lib\site-packages\nerfacc\volrend.py, line 137处：
添加"network_output":network_output,