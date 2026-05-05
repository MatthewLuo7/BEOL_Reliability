vm_offset_list="-6.0 -4.0 -2.0 0.0 2.0 4.0 6.0 8.0 10.0"
ll_space_list="5.5 6.5 7.5 8.5 9.5 10.5"
m_list="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0"

python -m exp.percolation_points.via2line_bt_fitting_gpr --vm-offset-list $vm_offset_list --ll-space $ll_space_list --via-dim-y 10.5 --via-dim-z 21.0 --line-dim-x 10.5 --line-dim-y 21.0 --line-dim-z 21.0 -r 0.45 --m-list $m_list -rN 2.0 --chunk-size 4 --cpu-num 64 --reload-breakdown-time