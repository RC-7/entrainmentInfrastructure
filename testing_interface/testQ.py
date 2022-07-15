from operator import mod
from Q_learning_interface import QLearningInterface

model_parameters = {
    "states":  ['up_24', 'down_24', 'up_27', 'down_27'], 
    'actions': ['24', '27'],
    "epsilon": 1,
    "learning_rate": 0.2,
    "discount_factor": 0.4,
    "step": 0
}
q_learn = QLearningInterface(model_parameters=None, model_path='models/', model_name = 'bciAgent')
# q_lean.save_model()
q_learn.update_entrainment('up')
q_learn.bellmans('up_24', 0.2)

# q_learn.save_model()