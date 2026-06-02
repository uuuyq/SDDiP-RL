#  RL 预测步长 eta

state：
action：
step：


Observation:
--------------------------------
log_gap
iteration_ratio

||w||²

gap_improve_last1
gap_improve_last2
gap_improve_last3

eta_current

recent_serious_ratio
--------------------------------

Action:
--------------------------------
a ∈ [-1,1]
--------------------------------

Update:
--------------------------------
η_new = η_old * exp(0.5*a)
--------------------------------

Reward:
--------------------------------
保持不变
--------------------------------